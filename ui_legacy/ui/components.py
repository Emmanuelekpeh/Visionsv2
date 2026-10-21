import pygame
from ui.theme import Theme

class Button:
    def __init__(self, x, y, w, h, text, callback=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.hovered = False
        self.disabled = False

    def draw(self, surface):
        color = Theme.ACCENT_HOVER if self.hovered and not self.disabled else Theme.ACCENT
        if self.disabled:
            color = Theme.BG_PANEL_LIGHT
            
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, Theme.BORDER, self.rect, 1, border_radius=4)
        
        text_color = Theme.TEXT_DIM if self.disabled else Theme.TEXT_MAIN
        surf = Theme.font_md.render(self.text, True, text_color)
        tr = surf.get_rect(center=self.rect.center)
        surface.blit(surf, tr)

    def handle_event(self, event):
        if self.disabled:
            return False
            
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.hovered:
                if self.callback:
                    self.callback()
                return True
        return False

class CycleButton(Button):
    def __init__(self, x, y, w, h, label, options, callback=None):
        self.label = label
        self.options = options
        self.current_idx = 0
        text = f"{self.label}: {self.options[self.current_idx]}"
        super().__init__(x, y, w, h, text, self._cycle)
        self.external_callback = callback
        
    def _cycle(self):
        self.current_idx = (self.current_idx + 1) % len(self.options)
        self.text = f"{self.label}: {self.options[self.current_idx]}"
        if self.external_callback:
            self.external_callback(self.options[self.current_idx])
            
    def get_value(self):
        return self.options[self.current_idx]
        
    def set_value(self, value):
        if value in self.options:
            self.current_idx = self.options.index(value)
            self.text = f"{self.label}: {self.options[self.current_idx]}"

class TextBox:
    @property
    def text(self):
        return self._text
        
    @text.setter
    def text(self, value):
        self._text = str(value)
        self.cursor_pos = len(self._text)

    def __init__(self, x, y, w, h, text=""):
        self.rect = pygame.Rect(x, y, w, h)
        self._text = str(text)
        self.active = False
        self.scroll_y = 0
        self.font = Theme.font_md
        self.line_height = self.font.get_linesize()
        self.padding = 10
        self.cursor_pos = len(self._text)

    def _wrap_text(self, text, max_width):
        lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split(' ')
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                if self.font.size(test_line)[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            if not paragraph:
                lines.append('')
        return lines

    def draw(self, surface):
        color = Theme.ACCENT if self.active else Theme.BORDER
        pygame.draw.rect(surface, Theme.BG_DARK, self.rect)
        pygame.draw.rect(surface, color, self.rect, 1)
        
        # Create a subsurface/clip rect for the text to prevent spilling
        clip_rect = self.rect.inflate(-self.padding * 2, -self.padding * 2)
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)
        
        # Insert cursor for rendering
        if self.active:
            display_text = self._text[:self.cursor_pos] + "|" + self._text[self.cursor_pos:]
        else:
            display_text = self._text
            
        lines = self._wrap_text(display_text, clip_rect.w)
        
        # Calculate total height
        total_height = len(lines) * self.line_height
        
        # Clamp scroll
        max_scroll = max(0, total_height - clip_rect.h)
        self.scroll_y = max(0, min(self.scroll_y, max_scroll))
        
        y_offset = clip_rect.y - self.scroll_y
        for line in lines:
            if y_offset + self.line_height > clip_rect.y and y_offset < clip_rect.bottom:
                surf = self.font.render(line, True, Theme.TEXT_MAIN)
                surface.blit(surf, (clip_rect.x, y_offset))
            y_offset += self.line_height
            
        surface.set_clip(old_clip)
        
        # Draw scrollbar if needed
        if total_height > clip_rect.h:
            sb_rect = pygame.Rect(self.rect.right - 8, self.rect.y + 2, 6, self.rect.h - 4)
            pygame.draw.rect(surface, Theme.BG_PANEL, sb_rect)
            
            thumb_h = max(20, int((clip_rect.h / total_height) * sb_rect.h))
            thumb_y = sb_rect.y + int((self.scroll_y / max_scroll) * (sb_rect.h - thumb_h)) if max_scroll > 0 else sb_rect.y
            thumb_rect = pygame.Rect(sb_rect.x, thumb_y, sb_rect.w, thumb_h)
            pygame.draw.rect(surface, Theme.TEXT_DIM, thumb_rect, border_radius=3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.active = self.rect.collidepoint(event.pos)
            elif event.button == 4 and self.rect.collidepoint(event.pos): # Scroll up
                self.scroll_y -= self.line_height * 2
            elif event.button == 5 and self.rect.collidepoint(event.pos): # Scroll down
                self.scroll_y += self.line_height * 2
                
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor_pos = min(len(self._text), self.cursor_pos + 1)
            elif event.key == pygame.K_UP:
                # Basic up: jump back 50 chars as a rough approximation
                self.cursor_pos = max(0, self.cursor_pos - 50)
            elif event.key == pygame.K_DOWN:
                # Basic down: jump forward 50 chars
                self.cursor_pos = min(len(self._text), self.cursor_pos + 50)
            elif event.key == pygame.K_RETURN:
                self._text = self._text[:self.cursor_pos] + '\n' + self._text[self.cursor_pos:]
                self.cursor_pos += 1
            elif event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self._text = self._text[:self.cursor_pos-1] + self._text[self.cursor_pos:]
                    self.cursor_pos -= 1
            elif event.key == pygame.K_DELETE:
                if self.cursor_pos < len(self._text):
                    self._text = self._text[:self.cursor_pos] + self._text[self.cursor_pos+1:]
            else:
                if event.unicode:
                    self._text = self._text[:self.cursor_pos] + event.unicode + self._text[self.cursor_pos:]
                    self.cursor_pos += len(event.unicode)
