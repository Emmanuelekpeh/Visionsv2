import pygame
import sys
import os
from PIL import Image
from ui.theme import Theme
from ui.components import Button, TextBox, CycleButton
from project.manager import ProjectManager
from backend.adapters import DiffusionBackend, OllamaBackend, VoiceBackend, AnimaticBackend, DistillationBackend
from pathlib import Path
import json

class App:
    def __init__(self):
        pygame.init()
        Theme.init()
        self.width, self.height = 1280, 800
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("LOCAL AI VIDEO STUDIO")
        self.clock = pygame.time.Clock()
        
        self.project = ProjectManager()
        self.active_shot_idx = 0
        
        # Backends
        self.diffusion = DiffusionBackend()
        self.ollama = OllamaBackend()
        self.voice = VoiceBackend()
        self.animatic = AnimaticBackend(self.voice, self.diffusion)
        self.distiller = DistillationBackend()
        
        # Playback state
        pygame.mixer.init()
        self.is_playing_animatic = False
        self.animatic_shot_start_time = 0
        self.current_animatic_idx = 0
        self.animatic_image_student = None
        self.animatic_image_teacher = None
        self.animatic_status = ""
        
        # Timeline state
        self.timeline_playhead = 0.0
        self.is_playing_timeline = False
        self.timeline_last_tick = 0
        self.timeline_status = ""
        
        self.state = "IDLE"
        self.mode = "CREATE" # CREATE, WORLD, STORYBOARD, ANIMATIC, PRODUCTION, TIMELINE
        self.current_image_surface = None
        
        self._build_ui()

    def _build_ui(self):
        # Top Navigation Tabs
        self.tabs = ["CREATE", "WORLD", "STORYBOARD", "ANIMATIC", "PRODUCTION", "TIMELINE"]
        self.tab_buttons = []
        tab_w = 120
        for i, tab in enumerate(self.tabs):
            btn = Button(20 + i * (tab_w + 10), 50, tab_w, 30, tab, lambda t=tab: self.set_mode(t))
            self.tab_buttons.append(btn)
            
        # CREATE Mode UI
        self.idea_box = TextBox(20, 130, self.width - 40, 60, self.project.state.get("idea", ""))
        self.btn_world_select = CycleButton(20, 200, 200, 30, "World", ["None", "The Deep", "Cyber City", "Surreal Space"])
        self.btn_build_concept = Button(230, 200, 150, 30, "DEVELOP IDEA", self.generate_concept)
        
        # Concept display (read-only for now, just a big text box)
        concept_str = json.dumps(self.project.state.get("concept", {}), indent=2)
        self.concept_box = TextBox(20, 240, (self.width - 60) // 3, 400, concept_str)
        
        self.btn_build_structure = Button(20 + (self.width - 60) // 3 + 10, 200, 180, 30, "GENERATE STRUCTURE", self.generate_structure)
        struct_str = json.dumps(self.project.state.get("structure", []), indent=2)
        self.structure_box = TextBox(20 + (self.width - 60) // 3 + 10, 240, (self.width - 60) // 3, 400, struct_str)
        
        self.btn_write_script = Button(20 + 2 * ((self.width - 60) // 3 + 10), 200, 150, 30, "WRITE SCRIPT", self.generate_script)
        self.script_box = TextBox(20 + 2 * ((self.width - 60) // 3 + 10), 240, (self.width - 60) // 3, 400, self.project.state.get("script", ""))
        
        # STORYBOARD Mode UI
        self.btn_gen_shots = Button(20, 100, 200, 30, "BUILD SHOT PLAN", self.generate_shots)
        self.btn_add_shot = Button(240, 100, 120, 30, "+ ADD SHOT", self.add_manual_shot)
        self.storyboard_scroll_y = 0
        
        # ANIMATIC Mode UI
        self.btn_build_animatic = Button(20, 100, 250, 30, "GENERATE SCRATCH AUDIO & SKETCHES", self.generate_animatic)
        self.btn_voice_select = CycleButton(280, 100, 220, 30, "Voice", ["en-US-ChristopherNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-JennyNeural"], self.update_voice_settings)
        self.btn_voice_rate = CycleButton(510, 100, 120, 30, "Speed", ["+15%", "+0%", "+10%", "+20%", "+25%", "-10%"], self.update_voice_settings)
        
        self.btn_play_animatic = Button(20, 560, 100, 30, "▶ PLAY", self.play_animatic)
        self.btn_stop_animatic = Button(130, 560, 100, 30, "■ STOP", self.stop_animatic)
        self.btn_teach_student = Button(240, 560, 200, 30, "TEACH STUDENT (MD WINS)", self.teach_student)
        self.btn_student_wins = Button(450, 560, 150, 30, "STUDENT WINS!", self.student_wins)
        
        # PRODUCTION Mode UI
        self.btn_teacher_select = CycleButton(self.width * 0.2 + 20, 460, 250, 30, "Teacher", ["tiny-sd", "bk-sdm-tiny", "amused-256"], self.update_teacher_settings)
        self.prompt_box = TextBox(self.width * 0.2 + 20, 500, self.width * 0.4 - 40, 80, "")
        
        # Add Steps Input
        self.steps_box = TextBox(self.width * 0.2 + 20, 590, 80, 30, "15")
        
        self.btn_generate = Button(self.width * 0.2 + 110, 590, 150, 30, "GENERATE KEYFRAME", self.generate_keyframe)
        self.btn_approve = Button(self.width * 0.2 + 270, 590, 150, 30, "APPROVE", self.approve_keyframe)
        
        right_x = int(self.width * 0.2) + int(self.width * 0.4)
        right_w = self.width - right_x
        
        self.detail_boxes = {
            "duration": TextBox(right_x + 20, 160, right_w - 40, 30, ""),
            "action": TextBox(right_x + 20, 230, right_w - 40, 60, ""),
            "dialogue": TextBox(right_x + 20, 330, right_w - 40, 60, ""),
            "sound": TextBox(right_x + 20, 430, right_w - 40, 40, ""),
            "purpose": TextBox(right_x + 20, 510, right_w - 40, 40, "")
        }
        self.btn_save_details = Button(right_x + 20, 570, 150, 30, "SAVE DETAILS", self.save_shot_details)
        self.btn_delete_shot = Button(right_x + 190, 570, 150, 30, "DELETE SHOT", self.delete_current_shot)
        
        # TIMELINE Mode UI
        self.btn_play_timeline = Button(20, 560, 100, 30, "▶ PLAY", self.play_timeline)
        self.btn_stop_timeline = Button(130, 560, 100, 30, "■ PAUSE", self.stop_timeline)
        self.btn_export_video = Button(240, 560, 150, 30, "EXPORT MOVIE", self.export_timeline)
        self.btn_render_preview = Button(400, 560, 180, 30, "RENDER PREVIEW", self.render_timeline_preview)
        self.btn_temporal_model = CycleButton(590, 560, 200, 30, "Latent Flow", ["RNN", "Transformer"])
        
        self._load_current_shot_image()
        self._update_shot_ui()

    def set_mode(self, mode):
        self.mode = mode
        if mode == "PRODUCTION" and len(self.project.state["shots"]) > 0:
            self._update_shot_ui()
            self._load_current_shot_image()

    def update_voice_settings(self, _=None):
        self.voice.set_settings(self.btn_voice_select.get_value(), self.btn_voice_rate.get_value())

    def update_teacher_settings(self, _=None):
        teacher_id = self.btn_teacher_select.get_value()
        self.diffusion.set_teacher(teacher_id)

    def generate_concept(self):
        if self.state != "IDLE": return
        self.state = "GENERATING"
        self.project.state["idea"] = self.idea_box.text
        self.project.save()
        
        def on_complete(concept_dict):
            self.project.state["concept"] = concept_dict
            self.project.save()
            self.concept_box.text = json.dumps(concept_dict, indent=2)
            self.state = "IDLE"
            
        self.ollama.generate_concept(self.idea_box.text, on_complete)

    def generate_structure(self):
        if self.state != "IDLE": return
        self.state = "GENERATING"
        
        try:
            self.project.state["concept"] = json.loads(self.concept_box.text)
            self.project.save()
        except:
            pass
            
        def on_complete(structure_list):
            self.project.state["structure"] = structure_list
            self.project.save()
            self.structure_box.text = json.dumps(structure_list, indent=2)
            self.state = "IDLE"
            
        self.ollama.generate_structure(self.project.state["concept"], on_complete)

    def generate_script(self):
        if self.state != "IDLE": return
        self.state = "GENERATING"
        
        try:
            self.project.state["structure"] = json.loads(self.structure_box.text)
            self.project.save()
        except:
            pass
            
        def on_complete(script_text):
            self.project.state["script"] = script_text
            self.project.save()
            self.script_box.text = script_text
            self.state = "IDLE"
            
        self.ollama.generate_script(self.project.state["structure"], on_complete)

    def generate_animatic(self):
        if self.state != "IDLE": return
        self.state = "GENERATING_ANIMATIC"
        self.animatic_status = "Starting..."
        
        def progress_cb(msg):
            self.animatic_status = msg
            
        def on_complete(updated_shots):
            self.project.state["shots"] = updated_shots
            self.project.save()
            self.state = "IDLE"
            self.current_animatic_idx = 0
            self._load_animatic_image()
            
        self.animatic.build_animatic(self.project.state["shots"], str(self.project.project_dir), on_complete, progress_cb)

    def _load_animatic_image(self):
        if len(self.project.state["shots"]) == 0:
            self.animatic_image_student = None
            self.animatic_image_teacher = None
            return
            
        shot = self.project.state["shots"][self.current_animatic_idx]
        
        def load_scaled(path, target_w):
            if not path: return None
            try:
                img = pygame.image.load(path)
                scale = target_w / img.get_width()
                new_h = int(img.get_height() * scale)
                return pygame.transform.scale(img, (target_w, new_h))
            except:
                return None
                
        self.animatic_image_student = load_scaled(shot.get("sketch_student", ""), 310)
        
        # Priority: Final Keyframe > Teacher Sketch
        t_path = shot.get("keyframe") or shot.get("sketch_teacher", "")
        self.animatic_image_teacher = load_scaled(t_path, 310)

    def play_animatic(self):
        if len(self.project.state["shots"]) == 0: return
        self.is_playing_animatic = True
        self.current_animatic_idx = 0
        self._load_animatic_image()
        import time
        self.animatic_shot_start_time = time.time()
        self._play_current_animatic_audio()

    def stop_animatic(self):
        self.is_playing_animatic = False
        pygame.mixer.music.stop()

    def _play_current_animatic_audio(self):
        shot = self.project.state["shots"][self.current_animatic_idx]
        audio_file = shot.get("audio_file")
        if audio_file and os.path.exists(audio_file):
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

    def teach_student(self):
        if self.state != "IDLE" or len(self.project.state["shots"]) == 0: return
        
        shot = self.project.state["shots"][self.current_animatic_idx]
        teacher_path = shot.get("sketch_teacher")
        
        if teacher_path and os.path.exists(teacher_path):
            # Save the teacher's image to distillation dataset
            distill_dir = Path("dataset/distillation")
            distill_dir.mkdir(parents=True, exist_ok=True)
            import time
            import shutil
            ts = int(time.time() * 1000)
            img_name = f"distill_{ts}.png"
            
            shutil.copy2(teacher_path, distill_dir / img_name)
            
            manifest_path = distill_dir / "manifest.jsonl"
            with open(manifest_path, "a", encoding="utf-8") as f:
                record = {
                    "file_name": img_name,
                    "prompt": shot["prompt"],
                    "timestamp": ts
                }
                f.write(json.dumps(record) + "\n")
                
            self.animatic_status = "Saved teacher output for distillation!"
            
            # Trigger background distillation
            self.distiller.trigger_distillation(lambda success: print("Animatic distillation complete." if success else "Animatic distillation skipped."))
            
    def student_wins(self):
        if self.state != "IDLE" or len(self.project.state["shots"]) == 0: return
        self.animatic_status = "Student won! No distillation needed."

    def generate_shots(self):
        if self.state != "IDLE": return
        self.state = "GENERATING"
        
        self.project.state["script"] = self.script_box.text
        self.project.save()
        
        def on_complete(shots_list):
            self.project.state["shots"] = []
            for shot_data in shots_list:
                self.project.add_shot(shot_data)
            self.state = "IDLE"
            
        self.ollama.generate_shots(self.project.state["script"], on_complete)

    def _update_shot_ui(self):
        shot = self.project.get_shot(self.active_shot_idx)
        if shot:
            self.prompt_box.text = shot.get("prompt", "")
            self.detail_boxes["duration"].text = str(shot.get("duration", 3.0))
            self.detail_boxes["action"].text = shot.get("action", "")
            self.detail_boxes["dialogue"].text = shot.get("dialogue", "")
            self.detail_boxes["sound"].text = shot.get("sound", "")
            self.detail_boxes["purpose"].text = shot.get("purpose", "")

    def _load_current_shot_image(self):
        shot = self.project.get_shot(self.active_shot_idx)
        img_path = None
        if shot:
            img_path = shot.get("keyframe") or shot.get("sketch_teacher") or shot.get("sketch_student")
        
        if img_path and os.path.exists(img_path):
            try:
                img = pygame.image.load(img_path)
                panel_w = int(self.width * 0.4) - 40
                scale = panel_w / img.get_width()
                new_h = int(img.get_height() * scale)
                self.current_image_surface = pygame.transform.scale(img, (panel_w, new_h))
            except:
                self.current_image_surface = None
        else:
            self.current_image_surface = None

    def add_manual_shot(self):
        new_shot = {
            "purpose": "New Shot",
            "camera": "Wide",
            "duration": 3.0,
            "action": "Describe action here...",
            "dialogue": "",
            "sound": "",
            "prompt": "visual keywords here"
        }
        self.project.add_shot(new_shot)
        self.active_shot_idx = len(self.project.state["shots"]) - 1
        self.set_mode("PRODUCTION")

    def save_shot_details(self):
        shot = self.project.get_shot(self.active_shot_idx)
        if shot:
            try:
                dur = float(self.detail_boxes["duration"].text.strip())
            except:
                dur = 3.0
            self.project.update_shot(self.active_shot_idx, "duration", dur)
            self.project.update_shot(self.active_shot_idx, "action", self.detail_boxes["action"].text)
            self.project.update_shot(self.active_shot_idx, "dialogue", self.detail_boxes["dialogue"].text)
            self.project.update_shot(self.active_shot_idx, "sound", self.detail_boxes["sound"].text)
            self.project.update_shot(self.active_shot_idx, "purpose", self.detail_boxes["purpose"].text)
            self.project.update_shot(self.active_shot_idx, "prompt", self.prompt_box.text)
            self.project.save()

    def delete_current_shot(self):
        if 0 <= self.active_shot_idx < len(self.project.state["shots"]):
            self.project.state["shots"].pop(self.active_shot_idx)
            self.project.save()
            self.active_shot_idx = max(0, self.active_shot_idx - 1)
            self._update_shot_ui()
            self._load_current_shot_image()

    def generate_keyframe(self):
        if self.state != "IDLE": return
        self.state = "GENERATING_KEYFRAME"
        self.btn_generate.disabled = True
        
        # Update project with edited prompt
        self.project.update_shot(self.active_shot_idx, "prompt", self.prompt_box.text)
        
        try:
            steps = int(self.steps_box.text.strip())
        except:
            steps = 15
            
        def on_complete(pil_img):
            if pil_img:
                temp_path = self.project.project_dir / "temp_keyframe.png"
                pil_img.save(temp_path)
                
                img = pygame.image.load(str(temp_path))
                panel_w = int(self.width * 0.4) - 40
                scale = panel_w / img.get_width()
                new_h = int(img.get_height() * scale)
                self.current_image_surface = pygame.transform.scale(img, (panel_w, new_h))
                
            self.state = "IDLE"
            self.btn_generate.disabled = False
            
        self.diffusion.generate_keyframe(self.prompt_box.text, on_complete, num_inference_steps=steps)

    def approve_keyframe(self):
        if self.current_image_surface:
            shot = self.project.get_shot(self.active_shot_idx)
            final_path = self.project.project_dir / f"shot_{shot['id']}_keyframe.png"
            pygame.image.save(self.current_image_surface, str(final_path))
            self.project.update_shot(self.active_shot_idx, "keyframe", str(final_path))
            self.project.update_shot(self.active_shot_idx, "prompt", self.prompt_box.text)
            
            # Save to distillation dataset
            distill_dir = Path("dataset/distillation")
            distill_dir.mkdir(parents=True, exist_ok=True)
            import time
            ts = int(time.time() * 1000)
            img_name = f"distill_{ts}.png"
            pygame.image.save(self.current_image_surface, str(distill_dir / img_name))
            
            manifest_path = distill_dir / "manifest.jsonl"
            with open(manifest_path, "a", encoding="utf-8") as f:
                record = {
                    "file_name": img_name,
                    "prompt": self.prompt_box.text,
                    "timestamp": ts
                }
                f.write(json.dumps(record) + "\n")
                
            # Trigger background distillation
            self.distiller.trigger_distillation(lambda success: print("Production distillation complete." if success else "Production distillation skipped."))

    def draw_create(self):
        title = Theme.font_lg.render("CREATE VIDEO", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(title, (20, 100))
        
        self.idea_box.draw(self.screen)
        self.btn_world_select.draw(self.screen)
        self.btn_build_concept.draw(self.screen)
        self.btn_build_structure.draw(self.screen)
        self.btn_write_script.draw(self.screen)
        
        self.concept_box.draw(self.screen)
        self.structure_box.draw(self.screen)
        self.script_box.draw(self.screen)
        
        if self.state == "GENERATING":
            gen_surf = Theme.font_lg.render("OLLAMA IS THINKING...", True, Theme.STATUS_BUSY)
            self.screen.blit(gen_surf, (20, 460))

    def draw_animatic(self):
        title = Theme.font_lg.render("ANIMATIC", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(title, (20, 100))
        info = Theme.font_md.render("Preview timing and story flow before expensive generation.", True, Theme.TEXT_DIM)
        self.screen.blit(info, (20, 140))
        
        self.btn_build_animatic.draw(self.screen)
        self.btn_voice_select.draw(self.screen)
        self.btn_voice_rate.draw(self.screen)
        
        if self.state == "GENERATING_ANIMATIC":
            stat = Theme.font_lg.render(self.animatic_status, True, Theme.STATUS_BUSY)
            self.screen.blit(stat, (300, 100))
            return
            
        if len(self.project.state["shots"]) == 0:
            return
            
        # Video Player Area
        player_rect = pygame.Rect(20, 180, 640, 360)
        pygame.draw.rect(self.screen, Theme.BG_PANEL_LIGHT, player_rect)
        pygame.draw.rect(self.screen, Theme.BORDER, player_rect, 1)
        
        # Current Shot Info
        shot = self.project.state["shots"][self.current_animatic_idx]
        
        # Draw VS Images
        if self.animatic_image_student:
            ix = player_rect.x + 5
            iy = player_rect.y + (player_rect.h - self.animatic_image_student.get_height()) // 2
            self.screen.blit(self.animatic_image_student, (ix, iy))
            lbl_s = Theme.font_sm.render("STUDENT (Direct CNN)", True, Theme.TEXT_HIGHLIGHT)
            self.screen.blit(lbl_s, (ix + 5, iy + 5))
            
        if self.animatic_image_teacher:
            ix = player_rect.x + 325
            iy = player_rect.y + (player_rect.h - self.animatic_image_teacher.get_height()) // 2
            self.screen.blit(self.animatic_image_teacher, (ix, iy))
            
            label_text = "FINAL KEYFRAME" if shot.get("keyframe") else "TEACHER (Fast MD)"
            lbl_t = Theme.font_sm.render(label_text, True, Theme.TEXT_HIGHLIGHT)
            self.screen.blit(lbl_t, (ix + 5, iy + 5))
            
        if not self.animatic_image_student and not self.animatic_image_teacher:
            txt = Theme.font_md.render("NO SKETCHES", True, Theme.TEXT_DIM)
            self.screen.blit(txt, (player_rect.centerx - 40, player_rect.centery))
            
        # Controls
        self.btn_play_animatic.draw(self.screen)
        self.btn_stop_animatic.draw(self.screen)
        self.btn_teach_student.draw(self.screen)
        self.btn_student_wins.draw(self.screen)
        
        # Mini Timeline
        tl_y = 610
        tl_h = 60
        pygame.draw.rect(self.screen, Theme.BG_PANEL, (20, tl_y, self.width - 40, tl_h))
        pygame.draw.rect(self.screen, Theme.BORDER, (20, tl_y, self.width - 40, tl_h), 1)
        
        total_duration = sum(s.get("duration", 3.0) for s in self.project.state["shots"])
        if total_duration > 0:
            px_per_sec = (self.width - 40) / total_duration
            curr_x = 20
            for i, s in enumerate(self.project.state["shots"]):
                sw = s.get("duration", 3.0) * px_per_sec
                color = Theme.ACCENT if i == self.current_animatic_idx else Theme.BG_PANEL_LIGHT
                pygame.draw.rect(self.screen, color, (curr_x, tl_y, sw, tl_h))
                pygame.draw.rect(self.screen, Theme.BORDER, (curr_x, tl_y, sw, tl_h), 1)
                
                if sw > 40:
                    lbl = Theme.font_sm.render(f"S{i:02d}", True, Theme.TEXT_MAIN)
                    self.screen.blit(lbl, (curr_x + 5, tl_y + 5))
                curr_x += sw
                
            # Playhead
            if self.is_playing_animatic:
                import time
                elapsed = time.time() - self.animatic_shot_start_time
                shot_start_time = sum(s.get("duration", 3.0) for s in self.project.state["shots"][:self.current_animatic_idx])
                playhead_x = 20 + (shot_start_time + elapsed) * px_per_sec
                pygame.draw.line(self.screen, Theme.STATUS_READY, (playhead_x, tl_y), (playhead_x, tl_y + tl_h), 2)
        
        # Shot text
        y = 180
        lbl = Theme.font_md.render(f"SHOT {self.current_animatic_idx:02d}", True, Theme.ACCENT)
        self.screen.blit(lbl, (700, y))
        
        dur = Theme.font_sm.render(f"Duration: {shot.get('duration', 0):.1f}s", True, Theme.TEXT_DIM)
        self.screen.blit(dur, (700, y + 30))
        
        act = Theme.font_sm.render(f"Action: {shot.get('action', '')}", True, Theme.TEXT_MAIN)
        self.screen.blit(act, (700, y + 60))
        
        dial = Theme.font_sm.render(f"Dialogue: {shot.get('dialogue', '')}", True, Theme.TEXT_MAIN)
        self.screen.blit(dial, (700, y + 90))

    def draw_world(self):
        title = Theme.font_lg.render("WORLD CONTINUITY", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(title, (20, 100))
        info = Theme.font_md.render("Characters, Locations, and Objects will be tracked here.", True, Theme.TEXT_DIM)
        self.screen.blit(info, (20, 140))

    def draw_storyboard(self):
        title = Theme.font_lg.render("STORYBOARD", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(title, (20, 100))
        self.btn_gen_shots.draw(self.screen)
        self.btn_add_shot.draw(self.screen)
        
        if self.state == "GENERATING":
            gen_surf = Theme.font_lg.render("PLANNING SHOTS...", True, Theme.STATUS_BUSY)
            self.screen.blit(gen_surf, (240, 100))
            
        # Draw grid of shots with clipping and scrolling
        clip_rect = pygame.Rect(20, 150, self.width - 40, self.height - 170)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(clip_rect)
        
        x, y = 20, 150 - self.storyboard_scroll_y
        w, h = 220, 180
        for i, shot in enumerate(self.project.state["shots"]):
            if y + h > 150 and y < self.height:
                rect = pygame.Rect(x, y, w, h)
                pygame.draw.rect(self.screen, Theme.BG_PANEL_LIGHT, rect)
                pygame.draw.rect(self.screen, Theme.BORDER, rect, 1)
                
                lbl = Theme.font_md.render(f"SHOT {i:02d}", True, Theme.TEXT_HIGHLIGHT)
                self.screen.blit(lbl, (x + 10, y + 10))
                
                cam = Theme.font_sm.render(shot.get("camera", "Wide")[:25], True, Theme.ACCENT)
                self.screen.blit(cam, (x + 10, y + 30))
                
                img_path = shot.get("keyframe") or shot.get("sketch_teacher") or shot.get("sketch_student")
                if img_path and os.path.exists(img_path):
                    try:
                        img = pygame.image.load(img_path)
                        img = pygame.transform.scale(img, (w - 20, h - 80))
                        self.screen.blit(img, (x + 10, y + 50))
                    except:
                        desc = Theme.font_sm.render(shot.get("action", "")[:30] + "...", True, Theme.TEXT_DIM)
                        self.screen.blit(desc, (x + 10, y + 70))
                else:
                    desc = Theme.font_sm.render(shot.get("action", "")[:30] + "...", True, Theme.TEXT_DIM)
                    self.screen.blit(desc, (x + 10, y + 70))
                    
                purp = Theme.font_sm.render(shot.get("purpose", "")[:30], True, Theme.TEXT_DIM)
                self.screen.blit(purp, (x + 10, y + h - 25))
                
            x += w + 20
            if x + w > self.width - 20:
                x = 20
                y += h + 20
                
        self.screen.set_clip(old_clip)

    def draw_production(self):
        # Left Panel (Shot List)
        left_w = int(self.width * 0.2)
        pygame.draw.rect(self.screen, Theme.BG_PANEL, (0, 90, left_w, self.height - 130))
        pygame.draw.line(self.screen, Theme.BORDER, (left_w, 90), (left_w, self.height - 40))
        
        proj_title = Theme.font_md.render("PRODUCTION SHOTS", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(proj_title, (10, 100))
        
        y = 130
        for i, shot in enumerate(self.project.state["shots"]):
            color = Theme.ACCENT if i == self.active_shot_idx else Theme.TEXT_DIM
            shot_surf = Theme.font_md.render(f"Shot {i:02d}", True, color)
            self.screen.blit(shot_surf, (10, y))
            y += 30
            
        # Center Panel (Current Shot)
        center_w = int(self.width * 0.4)
        center_x = left_w
        
        shot = self.project.get_shot(self.active_shot_idx)
        if not shot:
            no_shot = Theme.font_lg.render("NO SHOTS AVAILABLE. GO TO STORYBOARD.", True, Theme.TEXT_DIM)
            self.screen.blit(no_shot, (center_x + 20, 100))
            return
            
        shot_title = Theme.font_lg.render(f"SHOT {self.active_shot_idx:02d}: {shot.get('camera', '')}", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(shot_title, (center_x + 20, 100))
        
        # Image area
        img_rect = pygame.Rect(center_x + 20, 140, center_w - 40, center_w - 40)
        pygame.draw.rect(self.screen, Theme.BG_PANEL_LIGHT, img_rect)
        
        if self.current_image_surface:
            ix = img_rect.x + (img_rect.w - self.current_image_surface.get_width()) // 2
            iy = img_rect.y + (img_rect.h - self.current_image_surface.get_height()) // 2
            self.screen.blit(self.current_image_surface, (ix, iy))
            
            if shot.get("keyframe"):
                app_surf = Theme.font_sm.render("✓ APPROVED", True, Theme.STATUS_READY)
                self.screen.blit(app_surf, (img_rect.x + 10, img_rect.y + 10))
        else:
            no_img = Theme.font_md.render("NO KEYFRAME", True, Theme.TEXT_DIM)
            self.screen.blit(no_img, (img_rect.centerx - 50, img_rect.centery))
            
        if self.state == "GENERATING_KEYFRAME":
            gen_surf = Theme.font_lg.render("GENERATING...", True, Theme.STATUS_BUSY)
            self.screen.blit(gen_surf, (img_rect.centerx - 70, img_rect.centery + 30))
            
        # Controls
        self.btn_teacher_select.draw(self.screen)
        self.prompt_box.draw(self.screen)
        
        lbl_steps = Theme.font_sm.render("Steps:", True, Theme.TEXT_DIM)
        self.screen.blit(lbl_steps, (self.width * 0.2 + 20, 570))
        self.steps_box.draw(self.screen)
        
        self.btn_generate.draw(self.screen)
        self.btn_approve.draw(self.screen)
        
        # Right Panel (Preview / Details)
        right_w = self.width - (left_w + center_w)
        right_x = left_w + center_w
        pygame.draw.line(self.screen, Theme.BORDER, (right_x, 90), (right_x, self.height - 40))
        
        prev_title = Theme.font_md.render("SHOT DETAILS", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(prev_title, (right_x + 20, 100))
        
        labels = [
            ("DURATION (sec)", 140),
            ("ACTION", 210),
            ("DIALOGUE / NARRATION", 310),
            ("SOUND / MUSIC", 410),
            ("PURPOSE / CAMERA", 490)
        ]
        for text, y in labels:
            lbl = Theme.font_sm.render(text, True, Theme.TEXT_DIM)
            self.screen.blit(lbl, (right_x + 20, y))
            
        for box in self.detail_boxes.values():
            box.draw(self.screen)
            
        self.btn_save_details.draw(self.screen)
        self.btn_delete_shot.draw(self.screen)

    def draw_timeline(self):
        title = Theme.font_lg.render("TIMELINE ASSEMBLY", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(title, (20, 100))
        
        if self.timeline_status:
            stat = Theme.font_md.render(self.timeline_status, True, Theme.STATUS_BUSY)
            self.screen.blit(stat, (320, 105))
            
        shots = self.project.state["shots"]
        if not shots:
            info = Theme.font_md.render("No shots available. Build shots in Storyboard first.", True, Theme.TEXT_DIM)
            self.screen.blit(info, (20, 140))
            return
            
        # Draw Preview Player
        player_rect = pygame.Rect(20, 150, 640, 360)
        pygame.draw.rect(self.screen, Theme.BG_PANEL_LIGHT, player_rect)
        pygame.draw.rect(self.screen, Theme.BORDER, player_rect, 1)
        
        # Calculate global times
        total_dur = sum(s.get("duration", 3.0) for s in shots)
        
        # Find current shot based on playhead
        current_t = 0
        active_shot = None
        active_shot_start_t = 0
        for s in shots:
            dur = s.get("duration", 3.0)
            if current_t <= self.timeline_playhead < current_t + dur:
                active_shot = s
                active_shot_start_t = current_t
                break
            current_t += dur
            
        if not active_shot and shots:
            active_shot = shots[-1]
            active_shot_start_t = total_dur - active_shot.get("duration", 3.0)
            
        # Render active shot
        if active_shot:
            local_time = self.timeline_playhead - active_shot_start_t
            frame_idx = int(local_time * 12) # Assuming 12fps for preview
            
            preview_path = self.project.project_dir / "preview" / f"shot_{active_shot.get('id', 0)}" / f"frame_{frame_idx:04d}.jpg"
            img_path = str(preview_path) if preview_path.exists() else (active_shot.get("keyframe") or active_shot.get("sketch_teacher") or active_shot.get("sketch_student"))
            
            if img_path and os.path.exists(img_path):
                try:
                    img = pygame.image.load(img_path)
                    scale = player_rect.h / img.get_height()
                    new_w = int(img.get_width() * scale)
                    img = pygame.transform.scale(img, (new_w, player_rect.h))
                    ix = player_rect.x + (player_rect.w - new_w) // 2
                    self.screen.blit(img, (ix, player_rect.y))
                except:
                    pass
            
            # Show overlay text
            txt = Theme.font_sm.render(f"Shot {active_shot.get('id', 0)}", True, Theme.TEXT_HIGHLIGHT)
            self.screen.blit(txt, (player_rect.x + 10, player_rect.y + 10))
            
        # Controls
        self.btn_play_timeline.draw(self.screen)
        self.btn_stop_timeline.draw(self.screen)
        self.btn_export_video.draw(self.screen)
        self.btn_render_preview.draw(self.screen)
        self.btn_temporal_model.draw(self.screen)
        
        # Draw Tracks
        track_y = 610
        track_h = 40
        pygame.draw.rect(self.screen, Theme.BG_PANEL, (20, track_y, self.width - 40, track_h * 2 + 10))
        
        lbl_v = Theme.font_sm.render("V1", True, Theme.TEXT_DIM)
        lbl_a = Theme.font_sm.render("A1", True, Theme.TEXT_DIM)
        self.screen.blit(lbl_v, (25, track_y + 10))
        self.screen.blit(lbl_a, (25, track_y + 10 + track_h + 5))
        
        if total_dur > 0:
            px_per_sec = (self.width - 80) / total_dur
            curr_x = 60
            for i, s in enumerate(shots):
                sw = s.get("duration", 3.0) * px_per_sec
                
                # Video block
                v_color = Theme.ACCENT if s.get("keyframe") else Theme.BG_PANEL_LIGHT
                pygame.draw.rect(self.screen, v_color, (curr_x, track_y, sw, track_h))
                pygame.draw.rect(self.screen, Theme.BORDER, (curr_x, track_y, sw, track_h), 1)
                if sw > 30:
                    lbl = Theme.font_sm.render(f"S{i}", True, Theme.TEXT_MAIN)
                    self.screen.blit(lbl, (curr_x + 5, track_y + 5))
                
                # Audio block
                a_color = Theme.STATUS_READY if s.get("audio_file") else Theme.BG_DARK
                pygame.draw.rect(self.screen, a_color, (curr_x, track_y + track_h + 5, sw, track_h))
                pygame.draw.rect(self.screen, Theme.BORDER, (curr_x, track_y + track_h + 5, sw, track_h), 1)
                
                curr_x += sw
                
            # Playhead line
            playhead_x = 60 + (self.timeline_playhead * px_per_sec)
            pygame.draw.line(self.screen, Theme.TEXT_HIGHLIGHT, (playhead_x, track_y - 10), (playhead_x, track_y + track_h * 2 + 10), 2)
            
        time_lbl = Theme.font_md.render(f"{self.timeline_playhead:.1f}s / {total_dur:.1f}s", True, Theme.TEXT_MAIN)
        self.screen.blit(time_lbl, (680, 150))
        
        # Helper text
        warn = Theme.font_sm.render("Click RENDER PREVIEW to generate T+1 motion frames.", True, Theme.TEXT_DIM)
        self.screen.blit(warn, (680, 180))
        warn2 = Theme.font_sm.render("Audio is only available in EXPORT MOVIE.", True, Theme.TEXT_DIM)
        self.screen.blit(warn2, (680, 200))

    def play_timeline(self):
        self.is_playing_timeline = True
        import time
        self.timeline_last_tick = time.time()
        
    def stop_timeline(self):
        self.is_playing_timeline = False
        
    def render_timeline_preview(self):
        if self.state != "IDLE": return
        self.state = "RENDERING"
        self.timeline_status = "Rendering T+1 frames for preview..."
        
        def _run_render(use_transformer_flag):
            try:
                from production_system import Assembler
                from PIL import Image
                import numpy as np
                assembler = Assembler(output_dir=str(self.project.project_dir), device="cpu", visuals=self.diffusion.visuals, use_transformer=use_transformer_flag)
                
                preview_dir = self.project.project_dir / "preview"
                preview_dir.mkdir(exist_ok=True)
                
                shots = self.project.state["shots"]
                for i, shot in enumerate(shots):
                    shot_id = shot.get('id', i)
                    shot_dir = preview_dir / f"shot_{shot_id}"
                    shot_dir.mkdir(exist_ok=True)
                    
                    img_path = shot.get("keyframe") or shot.get("sketch_teacher") or shot.get("sketch_student")
                    if not img_path or not os.path.exists(img_path):
                        continue
                        
                    next_img_path = img_path
                    if i < len(shots) - 1:
                        next_shot = shots[i+1]
                        ni = next_shot.get("keyframe") or next_shot.get("sketch_teacher") or next_shot.get("sketch_student")
                        if ni and os.path.exists(ni):
                            next_img_path = ni
                            
                    self.timeline_status = f"Rendering Preview: Shot {shot_id}..."
                    frames = assembler._generate_frames(img_path, next_img_path, shot.get("prompt", ""), shot.get("duration", 3.0), fps=12)
                    
                    for f_idx, f_arr in enumerate(frames):
                        Image.fromarray(f_arr).save(shot_dir / f"frame_{f_idx:04d}.jpg", quality=85)
                        
                self.timeline_status = "Preview render complete! Press Play."
            except Exception as e:
                self.timeline_status = f"Preview render failed: {e}"
                import traceback
                traceback.print_exc()
            self.state = "IDLE"
            
        import threading
        use_transformer_flag = (self.btn_temporal_model.get_value() == "Transformer")
        threading.Thread(target=_run_render, args=(use_transformer_flag,), daemon=True).start()

    def export_timeline(self):
        if self.state != "IDLE": return
        self.state = "EXPORTING"
        self.timeline_status = "Exporting T+1 Frame-by-Frame Video to workspace/final_movie.mp4..."
        
        def _run_export(use_transformer_flag):
            try:
                from moviepy.editor import ImageClip, ImageSequenceClip, AudioFileClip, concatenate_videoclips
                import numpy as np
                from production_system import Assembler
                
                # Initialize the Assembler (T+1 generator)
                # Pass self.diffusion.visuals to allow Teacher Interruption if drift occurs
                assembler = Assembler(output_dir=str(self.project.project_dir), device="cpu", visuals=self.diffusion.visuals, use_transformer=use_transformer_flag)
                
                clips = []
                shots = self.project.state["shots"]
                
                for i, shot in enumerate(shots):
                    img_path = shot.get("keyframe") or shot.get("sketch_teacher") or shot.get("sketch_student")
                    
                    if not img_path or not os.path.exists(img_path):
                        # Fallback dummy clip
                        img = np.zeros((128, 128, 3), dtype=np.uint8)
                        clip = ImageClip(img).set_duration(shot.get("duration", 3.0))
                    else:
                        # Find the next image for T+1 interpolation target
                        next_img_path = img_path
                        if i < len(shots) - 1:
                            next_shot = shots[i+1]
                            ni = next_shot.get("keyframe") or next_shot.get("sketch_teacher") or next_shot.get("sketch_student")
                            if ni and os.path.exists(ni):
                                next_img_path = ni
                                
                        self.timeline_status = f"Generating T+1 frames for Shot {shot.get('id', i)}..."
                        
                        # Check if preview cache exists to save time!
                        preview_dir = self.project.project_dir / "preview" / f"shot_{shot.get('id', i)}"
                        if preview_dir.exists() and len(list(preview_dir.glob("*.jpg"))) > 0:
                            import glob
                            frame_files = sorted([str(p) for p in preview_dir.glob("*.jpg")])
                            clip = ImageSequenceClip(frame_files, fps=12)
                        else:
                            # Use the Assembler to generate intermediate frames
                            frames = assembler._generate_frames(img_path, next_img_path, shot.get("prompt", ""), shot.get("duration", 3.0), fps=12) # 12fps for reasonable CPU time
                            
                            if frames:
                                clip = ImageSequenceClip(frames, fps=12)
                            else:
                                clip = ImageClip(img_path).set_duration(shot.get("duration", 3.0))
                        
                    if shot.get("audio_file") and os.path.exists(shot.get("audio_file")):
                        audio = AudioFileClip(shot.get("audio_file"))
                        clip = clip.set_audio(audio)
                        
                    clips.append(clip)
                    
                if clips:
                    self.timeline_status = "Compositing video and audio..."
                    final = concatenate_videoclips(clips, method="compose")
                    out_path = str(self.project.project_dir / "final_movie.mp4")
                    final.write_videofile(out_path, fps=12, logger=None)
                    self.timeline_status = f"Exported: {out_path}"
                else:
                    self.timeline_status = "No clips to export."
            except Exception as e:
                self.timeline_status = f"Export failed: {e}"
                import traceback
                traceback.print_exc()
            self.state = "IDLE"
            
        import threading
        use_transformer_flag = (self.btn_temporal_model.get_value() == "Transformer")
        threading.Thread(target=_run_export, args=(use_transformer_flag,), daemon=True).start()

    def draw(self):
        self.screen.fill(Theme.BG_DARK)
        
        # Top Bar
        pygame.draw.rect(self.screen, Theme.BG_PANEL, (0, 0, self.width, 40))
        title = Theme.font_lg.render("LOCAL AI VIDEO STUDIO", True, Theme.TEXT_HIGHLIGHT)
        self.screen.blit(title, (20, 10))
        
        status = Theme.font_sm.render("● LOCAL / CPU", True, Theme.STATUS_READY)
        self.screen.blit(status, (self.width - 150, 15))
        
        # Tabs
        pygame.draw.rect(self.screen, Theme.BG_PANEL_LIGHT, (0, 40, self.width, 50))
        for btn in self.tab_buttons:
            btn.draw(self.screen)
            # Highlight active tab
            if btn.text == self.mode:
                pygame.draw.line(self.screen, Theme.ACCENT, (btn.rect.x, btn.rect.bottom + 5), (btn.rect.right, btn.rect.bottom + 5), 3)
        pygame.draw.line(self.screen, Theme.BORDER, (0, 90), (self.width, 90))
        
        # Mode Content
        if self.mode == "CREATE":
            self.draw_create()
        elif self.mode == "WORLD":
            self.draw_world()
        elif self.mode == "STORYBOARD":
            self.draw_storyboard()
        elif self.mode == "ANIMATIC":
            self.draw_animatic()
        elif self.mode == "PRODUCTION":
            self.draw_production()
        elif self.mode == "TIMELINE":
            self.draw_timeline()
            
        # Model Status (Bottom edge)
        pygame.draw.rect(self.screen, Theme.BG_DARK, (0, self.height - 30, self.width, 30))
        pygame.draw.line(self.screen, Theme.BORDER, (0, self.height - 30), (self.width, self.height - 30))
        
        status_text = "OLLAMA ●   MOBILE DIFF ●   T+1 ●   VOICE ●"
        stat_surf = Theme.font_sm.render(status_text, True, Theme.TEXT_DIM)
        self.screen.blit(stat_surf, (20, self.height - 22))

    def handle_shot_click(self, pos):
        # Handle clicking on shot list in PRODUCTION mode
        if self.mode == "PRODUCTION":
            left_w = int(self.width * 0.2)
            if pos[0] < left_w and pos[1] > 130:
                idx = (pos[1] - 130) // 30
                if 0 <= idx < len(self.project.state["shots"]):
                    self.active_shot_idx = idx
                    self._update_shot_ui()
                    self._load_current_shot_image()
                    
        # Handle clicking on storyboard grid
        if self.mode == "STORYBOARD":
            x, y = 20, 150 - self.storyboard_scroll_y
            w, h = 220, 180
            for i in range(len(self.project.state["shots"])):
                rect = pygame.Rect(x, y, w, h)
                if rect.collidepoint(pos):
                    self.active_shot_idx = i
                    self.set_mode("PRODUCTION")
                    return
                x += w + 20
                if x + w > self.width - 20:
                    x = 20
                    y += h + 20

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self._build_ui()
                    
                for btn in self.tab_buttons:
                    btn.handle_event(event)
                    
                if self.mode == "CREATE":
                    self.idea_box.handle_event(event)
                    self.btn_world_select.handle_event(event)
                    self.concept_box.handle_event(event)
                    self.structure_box.handle_event(event)
                    self.script_box.handle_event(event)
                    self.btn_build_concept.handle_event(event)
                    self.btn_build_structure.handle_event(event)
                    self.btn_write_script.handle_event(event)
                elif self.mode == "STORYBOARD":
                    self.btn_gen_shots.handle_event(event)
                    self.btn_add_shot.handle_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self.handle_shot_click(event.pos)
                        elif event.button == 4: # Scroll up
                            self.storyboard_scroll_y = max(0, self.storyboard_scroll_y - 40)
                        elif event.button == 5: # Scroll down
                            self.storyboard_scroll_y += 40
                elif self.mode == "ANIMATIC":
                    self.btn_build_animatic.handle_event(event)
                    self.btn_voice_select.handle_event(event)
                    self.btn_voice_rate.handle_event(event)
                    self.btn_play_animatic.handle_event(event)
                    self.btn_stop_animatic.handle_event(event)
                    self.btn_teach_student.handle_event(event)
                    self.btn_student_wins.handle_event(event)
                    
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            if self.is_playing_animatic:
                                self.stop_animatic()
                            else:
                                self.play_animatic()
                        elif event.key == pygame.K_RIGHT:
                            if self.current_animatic_idx < len(self.project.state["shots"]) - 1:
                                self.current_animatic_idx += 1
                                self._load_animatic_image()
                        elif event.key == pygame.K_LEFT:
                            if self.current_animatic_idx > 0:
                                self.current_animatic_idx -= 1
                                self._load_animatic_image()
                elif self.mode == "TIMELINE":
                    self.btn_play_timeline.handle_event(event)
                    self.btn_stop_timeline.handle_event(event)
                    self.btn_export_video.handle_event(event)
                    self.btn_render_preview.handle_event(event)
                    self.btn_temporal_model.handle_event(event)
                    
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        if self.is_playing_timeline:
                            self.stop_timeline()
                        else:
                            self.play_timeline()
                            
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        track_y = 610
                        track_h = 40
                        if 60 <= event.pos[0] <= self.width - 20 and track_y <= event.pos[1] <= track_y + track_h * 2 + 10:
                            total_dur = sum(s.get("duration", 3.0) for s in self.project.state["shots"])
                            if total_dur > 0:
                                px_per_sec = (self.width - 80) / total_dur
                                self.timeline_playhead = (event.pos[0] - 60) / px_per_sec
                                self.timeline_playhead = max(0, min(self.timeline_playhead, total_dur))
                elif self.mode == "PRODUCTION":
                    self.btn_teacher_select.handle_event(event)
                    self.prompt_box.handle_event(event)
                    self.steps_box.handle_event(event)
                    for box in self.detail_boxes.values():
                        box.handle_event(event)
                    self.btn_generate.handle_event(event)
                    self.btn_approve.handle_event(event)
                    self.btn_save_details.handle_event(event)
                    self.btn_delete_shot.handle_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.handle_shot_click(event.pos)
                
            self.draw()
            
            # Playback logic
            if self.is_playing_animatic:
                import time
                now = time.time()
                elapsed = now - self.animatic_shot_start_time
                shot = self.project.state["shots"][self.current_animatic_idx]
                if elapsed >= shot.get("duration", 3.0):
                    self.current_animatic_idx += 1
                    if self.current_animatic_idx >= len(self.project.state["shots"]):
                        self.is_playing_animatic = False
                        self.current_animatic_idx = 0
                        pygame.mixer.music.stop()
                        self._load_animatic_image()
                    else:
                        self.animatic_shot_start_time = time.time()
                        self._load_animatic_image()
                        self._play_current_animatic_audio()
                        
            if self.is_playing_timeline:
                import time
                now = time.time()
                dt = now - self.timeline_last_tick
                self.timeline_last_tick = now
                self.timeline_playhead += dt
                
                total_dur = sum(s.get("duration", 3.0) for s in self.project.state["shots"])
                if self.timeline_playhead >= total_dur:
                    self.timeline_playhead = 0
                    self.is_playing_timeline = False
                    
            pygame.display.flip()
            self.clock.tick(30)
            
        pygame.quit()
        sys.exit()
