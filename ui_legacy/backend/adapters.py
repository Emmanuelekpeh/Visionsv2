import os
import json
import threading
from PIL import Image
from typing import Dict, Any, Callable

# Import existing logic
import ollama
from production_system import VisualGenerator, VoiceModule
from distill_student import distill_student_model

class DistillationBackend:
    def __init__(self, device="cpu"):
        self.device = device
        self.is_training = False
        
    def trigger_distillation(self, callback: Callable[[bool], None] = None):
        """Runs a quick training pass on the approved frames in the background."""
        if self.is_training:
            print("Distillation already running. Skipping trigger.")
            if callback: callback(False)
            return
            
        def _run():
            self.is_training = True
            try:
                success = distill_student_model(device=self.device, epochs=5)
                if callback: callback(success)
            except Exception as e:
                print(f"Distillation failed: {e}")
                if callback: callback(False)
            finally:
                self.is_training = False
                
        threading.Thread(target=_run, daemon=True).start()

class DiffusionBackend:
    def __init__(self):
        self.visuals = None
        self.loaded = False
        self.teacher_id = "tiny-sd"
        
    def set_teacher(self, teacher_id: str):
        self.teacher_id = teacher_id
        # Force reload with new teacher
        self.loaded = False
        self.visuals = None
        
    def load(self):
        if not self.loaded:
            try:
                self.visuals = VisualGenerator(device="cpu", teacher_id=self.teacher_id)
                self.loaded = True
            except Exception as e:
                print(f"Failed to load VisualGenerator: {e}")

    def generate_keyframe(self, prompt: str, callback: Callable[[Image.Image], None], num_inference_steps: int = 15):
        """Generates the final keyframe."""
        def _run():
            self.load()
            if self.visuals:
                img = self.visuals.generate_keyframe(prompt, num_inference_steps=num_inference_steps)
                callback(img)
            else:
                callback(None)
        threading.Thread(target=_run, daemon=True).start()
        
    def generate_sketch(self, prompt: str, callback: Callable[[Image.Image], None]):
        """Generates a fast Direct CNN sketch for the animatic."""
        def _run():
            self.load()
            if self.visuals:
                img = self.visuals.generate_support_keyframe(prompt)
                callback(img)
            else:
                callback(None)
        threading.Thread(target=_run, daemon=True).start()
        
    def generate_1step_md(self, prompt: str, callback: Callable[[Image.Image], None]):
        """Generates a 1-step Mobile Diffusion image for clarity and training."""
        def _run():
            self.load()
            if self.visuals and self.visuals.core_model:
                img = self.visuals.core_model.generate_keyframe(prompt, num_inference_steps=1, guidance_scale=1.0)
                callback(img)
            else:
                callback(None)
        threading.Thread(target=_run, daemon=True).start()

class AnimaticBackend:
    def __init__(self, voice_module, diffusion_backend):
        self.voice = voice_module
        self.diffusion = diffusion_backend
        
    def build_animatic(self, shots, project_dir, callback, progress_callback):
        def _run():
            # 1. Generate Audio
            script_for_voice = []
            for shot in shots:
                text = shot.get("dialogue", "")
                if not text: text = shot.get("action", "")
                if not text: text = shot.get("prompt", "Silence.")
                script_for_voice.append({"text": text, "speaker": "Narrator"})
            
            progress_callback(f"Generating Audio ({self.voice.rate})...")
            timed_script = self.voice.module.generate_audio_for_script(
                script_for_voice, 
                default_voice=self.voice.voice_name, 
                rate=self.voice.rate
            )
            
            # 2. Generate VS Sketches (Student vs Teacher)
            self.diffusion.load()
            for i, shot in enumerate(shots):
                progress_callback(f"Generating VS Sketches {i+1}/{len(shots)}...")
                shot["audio_file"] = timed_script[i]["audio_file"]
                shot["duration"] = timed_script[i]["duration"]
                
                if self.diffusion.visuals:
                    # Student (Direct CNN)
                    img_s = self.diffusion.visuals.generate_support_keyframe(shot["prompt"])
                    path_s = os.path.join(project_dir, f"shot_{shot['id']}_student.png")
                    img_s.save(path_s)
                    shot["sketch_student"] = path_s
                    
                    # Teacher (Fast MD)
                    if self.diffusion.visuals.core_model:
                        # Testing shows 4 steps at CFG 4.0 provides a decent balance of speed (around 50s) and image coherency
                        img_t = self.diffusion.visuals.core_model.generate_keyframe(shot["prompt"], num_inference_steps=4, guidance_scale=4.0)
                        path_t = os.path.join(project_dir, f"shot_{shot['id']}_teacher.png")
                        img_t.save(path_t)
                        shot["sketch_teacher"] = path_t
                
            callback(shots)
        threading.Thread(target=_run, daemon=True).start()

class OllamaBackend:
    def __init__(self, model_name="qwen2.5:3b"):
        self.model_name = model_name
        self.available = False
        try:
            ollama.list()
            self.available = True
        except:
            pass
            
        self.aesthetic_prompt = """VISUAL & CREATIVE IDENTITY: "The Uncanny Documentary"
The visual language is observational, surreal, tactile and slightly uncanny.
Images should feel discovered rather than manufactured.
Treat strange subjects with the visual seriousness normally given to ordinary reality.

Favor imperfect, physical imagery over pristine digital perfection. Use soft detail, natural texture, filmic grain, subtle optical imperfections, restrained color and imperfect exposure when appropriate.
Composition should sometimes be awkward, asymmetrical, partially obstructed or unexpectedly framed. Do not force every image into conventional cinematic composition.
Use darkness, distance, shadows, reflections, environmental obstruction and partial visibility to preserve uncertainty. Do not reveal or explain strange subjects immediately.

Prefer environments that feel inhabited and physically real: ordinary rooms, streets, forests, oceans, laboratories, industrial spaces, homes and mundane landscapes containing something subtly impossible.
The strange element should coexist with normal reality rather than transforming the entire world into spectacle.

Favor human-scale observation, quiet reactions, hesitation, stillness and moments where nothing happens.
Use visual contradictions: beautiful and ugly, cute and disturbing, mundane and alien, organic and mechanical, enormous and tiny, funny and unsettling.
Color should generally be restrained rather than hyper-saturated, with unusual colors appearing selectively as visual events.

Avoid generic cinematic spectacle, glossy AI imagery, excessive sharpness, perfect skin, heroic compositions, constant camera movement, generic horror imagery, fantasy excess, gratuitous VHS effects and artificial "cinematic" filters.

The image should create curiosity before providing explanation. The audience should occasionally wonder: "What exactly am I looking at?"
CRITICAL: Do not make an image strange merely by adding strange objects. Make the relationship between ordinary things strange. Observe strangeness as a documentary would."""

    def generate_concept(self, idea: str, callback: Callable[[Dict[str, str]], None]):
        def _run():
            if not self.available:
                callback({
                    "title": "Ollama is not running",
                    "format": "Manual Entry",
                    "target_length": "N/A",
                    "core_question": "N/A",
                    "hook": "Please type your concept manually.",
                    "premise": "",
                    "escalation": "",
                    "payoff": "",
                    "visual_approach": "",
                    "voice": "",
                    "ending": ""
                })
                return
            try:
                import random
                model_to_use = random.choice([self.model_name, "phi3:latest"])
                print(f"Using {model_to_use} for concept generation...")
                
                prompt = f"{self.aesthetic_prompt}\n\nBased on this idea: '{idea}', generate a YouTube-native video concept. Output ONLY valid JSON with keys: 'title', 'format', 'target_length', 'hook', 'central_question', 'premise', 'escalation', 'turn', 'payoff', 'visual_approach', 'ending'."
                response = ollama.chat(model=model_to_use, messages=[{'role': 'user', 'content': prompt}])
                content = response['message']['content']
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    callback(json.loads(content[start:end]))
                else:
                    raise ValueError("No JSON found")
            except Exception as e:
                print(f"Ollama error: {e}")
                callback({"title": "Error generating.", "hook": str(e)})
                
        threading.Thread(target=_run, daemon=True).start()

    def generate_structure(self, concept: Dict[str, str], callback: Callable[[list], None]):
        def _run():
            if not self.available:
                callback([{"beat": "HOOK", "description": "Manual entry", "duration": 5}])
                return
            try:
                import random
                model_to_use = random.choice([self.model_name, "phi3:latest"])
                print(f"Using {model_to_use} for structure generation...")
                
                prompt = f"{self.aesthetic_prompt}\n\nBreak this concept down into a sequence of narrative beats (e.g., HOOK, CONTEXT, DISCOVERY, ESCALATION, TURN, PAYOFF) adhering to the documentary observation aesthetic described above. Output ONLY a JSON array of objects. Each object must have: 'beat' (name of the beat), 'description' (what happens), 'duration' (estimated seconds).\nConcept:\n{json.dumps(concept)}"
                response = ollama.chat(model=model_to_use, messages=[{'role': 'user', 'content': prompt}])
                content = response['message']['content']
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != 0:
                    callback(json.loads(content[start:end]))
                else:
                    raise ValueError("No JSON array found")
            except Exception as e:
                print(f"Ollama error: {e}")
                callback([])
                
        threading.Thread(target=_run, daemon=True).start()

    def generate_script(self, structure: list, callback: Callable[[str], None]):
        def _run():
            if not self.available:
                callback("Ollama is not running. Please type your script manually.")
                return
            try:
                import random
                model_to_use = random.choice([self.model_name, "phi3:latest"])
                print(f"Using {model_to_use} for script generation...")
                
                prompt = f"{self.aesthetic_prompt}\n\nWrite a short, visual, observational screenplay script based on these beats:\n{json.dumps(structure, indent=2)}\nKeep it understated, surreal, and grounded as an 'uncanny documentary'. Focus heavily on observation and restraint. Let silence and holds do the heavy lifting."
                response = ollama.chat(model=model_to_use, messages=[{'role': 'user', 'content': prompt}])
                callback(response['message']['content'].strip())
            except Exception as e:
                callback(f"Error: {e}")
                
        threading.Thread(target=_run, daemon=True).start()

    def generate_shots(self, script: str, callback: Callable[[list], None]):
        def _run():
            if not self.available:
                callback([
                    {"purpose": "Manual shot", "camera": "Wide", "duration": 3.0, "action": "", "prompt": ""}
                ])
                return
            try:
                import random
                model_to_use = random.choice([self.model_name, "phi3:latest"])
                print(f"Using {model_to_use} for shot planning...")
                
                prompt = f"{self.aesthetic_prompt}\n\nBreak this script into a storyboard of visual anchors (keyframes). Create a new shot ONLY when there is a major visual state change (location, composition, subject, major action).\nIMPORTANT: Apply the 'uncanny documentary' aesthetic heavily to the camera angles, framing, lighting, and action details.\nOutput ONLY a JSON array of objects. Each object must have: 'purpose', 'camera', 'duration' (number in seconds), 'action', 'dialogue', 'sound', 'prompt'.\n\nCRITICAL RULE FOR 'prompt': The prompt field MUST describe the physical contents of the shot FIRST (e.g., 'abandoned theater, dusty stage, distant figure in shadow'), AND THEN end with a consistent global visual fingerprint (e.g., ', filmic grain, muted colors, soft focus'). Do not just output the aesthetic fingerprint! You must describe the subject of the shot!\n\nScript:\n{script}"
                response = ollama.chat(model=model_to_use, messages=[{'role': 'user', 'content': prompt}])
                content = response['message']['content']
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != 0:
                    callback(json.loads(content[start:end]))
                else:
                    raise ValueError("No JSON array found")
            except Exception as e:
                print(f"Ollama error: {e}")
                callback([])
                
        threading.Thread(target=_run, daemon=True).start()

class VoiceBackend:
    def __init__(self):
        self.module = VoiceModule(output_dir="workspace/audio")
        self.voice_name = "en-US-ChristopherNeural"
        self.rate = "+15%"
        
    def set_settings(self, voice_name: str, rate: str):
        self.voice_name = voice_name
        self.rate = rate
        
    def generate(self, text: str, callback: Callable[[str], None]):
        def _run():
            # Using the existing script generation format for compatibility
            script = [{"speaker": "Narrator", "text": text}]
            res = self.module.generate_audio_for_script(script, default_voice=self.voice_name, rate=self.rate)
            if res and 'audio_file' in res[0]:
                callback(res[0]['audio_file'])
            else:
                callback(None)
                
        threading.Thread(target=_run, daemon=True).start()
