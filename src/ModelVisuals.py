# Importing pygame module
import pygame
from pygame.locals import *

pygame.init()

windowSurface = pygame.display.set_mode((600, 600))


def ClearScreen():
    windowSurface.fill((255, 255, 255))

def RenderStep():
    ClearScreen()
    pygame.draw.circle(windowSurface, (0, 255, 0), [300, 300], 170, 0)
    pygame.display.update()


