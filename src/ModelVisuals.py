# Importing pygame module
import pygame
from pygame.locals import *

def ClearScreen(windowSurface):
    windowSurface.fill((255, 255, 255));

def RenderStep(windowSurface):
    ClearScreen(windowSurface);
    pygame.draw.circle(windowSurface, (0, 255, 0), [300, 300], 170, 0);
    pygame.display.update();

def HandleEvent(event):
    #TODO
    pass;