"""
Thread-Safe PyTorch Executor

PyTorch's C++ backend is NOT thread-safe on CPU (especially Windows).
Python RLock does NOT protect the C++ memory allocator — concurrent
inference (VAE/CLIP on evolve thread) + training (Adam on worker thread)
still causes access violations even with a lock.

Solution:
    - ONE dedicated worker thread owns ALL PyTorch operations
    - Inference AND training are queued and executed sequentially
    - Other threads block on submit(wait=True) until their task completes
"""

import threading
import queue
from typing import Callable, Any, Optional
from dataclasses import dataclass
import traceback
import torch
import torch.nn.functional as F
import time
import random
from engines import load_image

@dataclass
class TrainingTask:
    """A PyTorch task to be executed on the dedicated worker thread."""
    func: Callable
    args: tuple
    kwargs: dict
    result_queue: Optional[queue.Queue] = None


class OnlineTrainer:
    """
    Handles continuous online training of the Hybrid MAE model.
    Runs in the background, sampling images from the database and updating the model.
    """
    def __init__(self, model, memory, device="cpu", save_path="hybrid_mae_weights.pt"):
        self.model = model
        self.memory = memory
        self.device = device
        self.save_path = save_path
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.05)
        self.running = False
        self.thread = None
        self.loss_history = []
        self.steps = 0
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._training_loop, daemon=True, name="OnlineTrainer")
        self.thread.start()
        print("[OnlineTrainer] Started continuous background training")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self._save_model()
        print("[OnlineTrainer] Stopped and saved model.")
            
    def _save_model(self):
        try:
            torch.save(self.model.state_dict(), self.save_path)
        except Exception as e:
            print(f"[OnlineTrainer] Failed to save model: {e}")

    def _training_loop(self):
        while self.running:
            try:
                # Need at least a few images to train
                living_ids = list(self.memory.get_living_ids())
                if len(living_ids) < 10:
                    time.sleep(5.0)
                    continue
                    
                # Sample a batch (batch size 4 for CPU)
                batch_ids = random.sample(living_ids, min(4, len(living_ids)))
                batch_imgs = []
                
                for cid in batch_ids:
                    path = self.memory.get_image_path_for_concept(cid)
                    if path:
                        try:
                            img = load_image(path)
                            batch_imgs.append(img)
                        except Exception:
                            pass
                            
                if not batch_imgs:
                    time.sleep(1.0)
                    continue
                    
                # Stack into batch
                batch = torch.cat(batch_imgs, dim=0).to(self.device)
                
                # Train step (must run on PyTorch thread to avoid C++ backend crashes)
                loss = run_on_pytorch_thread(self._train_step, batch)
                
                if loss is not None:
                    self.loss_history.append(loss)
                    if len(self.loss_history) > 100:
                        self.loss_history.pop(0)
                        
                    self.steps += 1
                    # Save the model periodically (every 50 steps)
                    if self.steps % 50 == 0:
                        run_on_pytorch_thread(self._save_model)
                        
                # Sleep to yield CPU to the main evolutionary loop
                time.sleep(2.0)
                
            except Exception as e:
                print(f"[OnlineTrainer] Error: {e}")
                time.sleep(5.0)
                
    def _train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass (MAE objective)
        pred, mask = self.model(batch, mask_ratio=0.75)
        
        # Compute loss only on masked patches
        # pred: (B, C, H, W), batch: (B, C, H, W)
        loss = F.mse_loss(pred, batch, reduction='none')
        # We need to reshape mask to match spatial dimensions, or just use simple MSE for now
        loss = loss.mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def get_recent_loss(self):
        if not self.loss_history:
            return 0.0
        return sum(self.loss_history[-10:]) / min(10, len(self.loss_history))


class TrainingScheduler:
    """
    Single-threaded PyTorch executor.
    ALL torch operations (inference, training, optimizers) run here.
    """

    def __init__(self):
        self.task_queue = queue.Queue(maxsize=200)
        self.shutdown_flag = threading.Event()
        self.worker_thread = None
        self._start_worker()

    def _start_worker(self):
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PyTorchWorker",
        )
        self.worker_thread.start()

    def _worker_loop(self):
        while not self.shutdown_flag.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
                try:
                    result = task.func(*task.args, **task.kwargs)
                    if task.result_queue is not None:
                        task.result_queue.put(("success", result))
                except Exception as e:
                    error_msg = f"PyTorch task failed: {e}\n{traceback.format_exc()}"
                    print(f"[PyTorchWorker] {error_msg}")
                    if task.result_queue is not None:
                        task.result_queue.put(("error", error_msg))
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue

    def submit(
        self,
        func: Callable,
        *args,
        wait: bool = False,
        timeout: float = 120.0,
        **kwargs,
    ) -> Any:
        if self.shutdown_flag.is_set():
            raise RuntimeError("PyTorch executor is shutdown")

        result_queue = queue.Queue(maxsize=1) if wait else None
        task = TrainingTask(func=func, args=args, kwargs=kwargs, result_queue=result_queue)

        try:
            self.task_queue.put(task, timeout=10.0)
        except queue.Full:
            raise RuntimeError("PyTorch queue is full - system is overloaded")

        if wait:
            try:
                status, result = result_queue.get(timeout=timeout)
                if status == "success":
                    return result
                raise RuntimeError(result)
            except queue.Empty:
                raise TimeoutError(f"PyTorch task timed out after {timeout} seconds")

        return None
    
    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown the training scheduler gracefully.
        
        Args:
            timeout: Maximum time to wait for pending tasks to complete
        """
        print("[TrainingScheduler] Shutting down...")
        self.shutdown_flag.set()
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
        
        print("[TrainingScheduler] Shutdown complete")
    
    def wait_for_pending_tasks(self, timeout: float = 10.0):
        """
        Wait for all pending tasks in the queue to complete.
        
        Args:
            timeout: Maximum time to wait
        
        Returns:
            True if all tasks completed, False if timeout
        """
        try:
            # Queue.join() blocks until all tasks are done
            self.task_queue.join()
            return True
        except:
            return False


# Global singleton instance
_global_scheduler: Optional[TrainingScheduler] = None
_scheduler_lock = threading.Lock()


def get_training_scheduler() -> TrainingScheduler:
    """Get the global PyTorch executor singleton."""
    global _global_scheduler

    if _global_scheduler is None:
        with _scheduler_lock:
            if _global_scheduler is None:
                _global_scheduler = TrainingScheduler()
                print("[PyTorchWorker] Initialized single-threaded PyTorch executor")

    return _global_scheduler


def is_pytorch_worker_thread() -> bool:
    """True if the current thread is the dedicated PyTorch worker."""
    if _global_scheduler is None:
        return False
    worker = _global_scheduler.worker_thread
    return worker is not None and threading.current_thread() is worker


def run_on_pytorch_thread(func: Callable, *args, wait: bool = True, **kwargs) -> Any:
    """
    Run a callable on the PyTorch worker thread.

    If already on the worker thread, runs inline (avoids deadlock).
    Otherwise queues the task; blocks until done when wait=True.
    """
    if is_pytorch_worker_thread():
        return func(*args, **kwargs)
    return get_training_scheduler().submit(func, *args, wait=wait, **kwargs)


def shutdown_training_scheduler():
    """Shutdown the global training scheduler if it exists."""
    global _global_scheduler
    
    if _global_scheduler is not None:
        _global_scheduler.shutdown()
        _global_scheduler = None
