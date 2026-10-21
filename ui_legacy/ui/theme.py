import pygame

class Theme:
    BG_DARK = (15, 15, 18)
    BG_PANEL = (25, 25, 30)
    BG_PANEL_LIGHT = (35, 35, 42)
    
    TEXT_MAIN = (220, 220, 220)
    TEXT_DIM = (120, 120, 130)
    TEXT_HIGHLIGHT = (255, 255, 255)
    
    ACCENT = (60, 120, 210)
    ACCENT_HOVER = (80, 140, 230)
    
    STATUS_READY = (50, 200, 100)
    STATUS_ERROR = (220, 60, 60)
    STATUS_BUSY = (220, 180, 50)
    
    BORDER = (45, 45, 55)
    
    @classmethod
    def init(cls):
        pygame.font.init()
        # Fallback to default font, but try to use a monospaced/technical font if possible
        font_name = pygame.font.match_font("consolas")
        if font_name:
            cls.font_sm = pygame.font.Font(font_name, 12)
            cls.font_md = pygame.font.Font(font_name, 16)
            cls.font_lg = pygame.font.Font(font_name, 24)
        else:
            cls.font_sm = pygame.font.Font(None, 14)
            cls.font_md = pygame.font.Font(None, 18)
            cls.font_lg = pygame.font.Font(None, 26)
