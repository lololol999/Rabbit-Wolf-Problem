import ModelVisuals
import pygame

FPS = 30                         

pygame.init()
windowSurface = pygame.display.set_mode((1000, 700));
clock = pygame.time.Clock()


windowHandler = ModelVisuals.WindowHandler(windowSurface)


running = True
while running:
    for event in pygame.event.get():
        windowHandler.HandleEvent(event);
        if event.type == pygame.QUIT:
            running = False;
        
    windowHandler.RenderStep(windowSurface)
    clock.tick(FPS) 

pygame.quit()