import ModelVisuals
import pygame


pygame.init();
windowSurface = pygame.display.set_mode((600, 600));

running = True;
while running:
    for event in pygame.event.get():
        ModelVisuals.HandleEvent(event);
        if event.type == pygame.QUIT:
            running = False;
        
    ModelVisuals.RenderStep(windowSurface);