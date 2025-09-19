# Importing pygame module
import pygame
from pygame.locals import *

import Colors as COLORS
import RabbitModel as model

class NeuralNetworkVisualizer:
    def __init__(self, startX, startY, width, height):
        self.start_x = startX
        self.start_y = startY
        self.drawing_space_width = width
        self.drawing_space_height = height
        self.layer_spacing = width // (len(model.net.layer_activations) + 1)
        self.neuron_radius = 15
        self.activations = model.net.layer_activations
        self.font = pygame.font.SysFont('Arial', 14)

    # Neuron shit
    def GetNeuronColor(self, activation):
        # Converts activation value to a color
        if activation > 0:
            # Green for positive values
            intensity = min(255, int(activation * 255))
            return (255 - intensity, 255, 255 - intensity)
        else:
            # Red for negative values
            intensity = min(255, int(-activation * 255))
            return (255, 255 - intensity, 255 - intensity)
        

    def update_activations(self):
        # Get the latest activations from the model
        self.activations = model.net.layer_activations

    def draw_neuron_connections(self, surface):
        for layer_idx in range(len(self.activations) - 1):
            current_layer = self.activations[layer_idx]
            next_layer = self.activations[layer_idx + 1]
            
            current_x = (layer_idx + 1) * self.layer_spacing
            next_x = (layer_idx + 2) * self.layer_spacing
            
            for i, current_activation in enumerate(current_layer):
                current_y = self.drawing_space_height * (i + 1) / (len(current_layer) + 1) + self.start_y
                
                for j, next_activation in enumerate(next_layer):
                    next_y = self.drawing_space_height * (j + 1) / (len(next_layer) + 1) + self.start_y
                    
                    # Calculate line properties based on activations
                    alpha = min(255, int(abs(current_activation) * 255))
                    width = max(1, int(abs(current_activation) * 3))
                    
                    # Create a surface for the line with alpha
                    line_surface = pygame.Surface((abs(next_x - current_x), abs(next_y - current_y) + 1), pygame.SRCALPHA)
                    
                    if current_activation > 0:
                        color = (0, 255, 0, alpha)  # Green for positive
                    else:
                        color = (255, 0, 0, alpha)  # Red for negative
                    
                    pygame.draw.line(line_surface, color, (0, 0), 
                                    (next_x - current_x, next_y - current_y), width)
                    
                    surface.blit(line_surface, (current_x, current_y))

    def draw_network(self, surface):
        # Draw connections first (behind neurons)
        self.draw_neuron_connections(surface)
        
        # Draw neurons
        for layer_idx, layer_activations in enumerate(self.activations):
            neuron_count = len(layer_activations)
            layer_x = (layer_idx + 1) * self.layer_spacing
            
            for neuron_idx in range(neuron_count):
                neuron_y = self.window_height * (neuron_idx + 1) / (neuron_count + 1)
                
                # Get activation value and convert to color
                activation = layer_activations[neuron_idx]
                color = self.get_neuron_color(activation)
                
                # Draw neuron
                pygame.draw.circle(surface, color, (layer_x, neuron_y), self.neuron_radius)
                pygame.draw.circle(surface, COLORS.BLACK, (layer_x, neuron_y), self.neuron_radius, 1)
                
                # Draw activation value
                if abs(activation) > 0.01:  # Only show significant activations
                    text = self.font.render(f"{activation:.2f}", True, COLORS.BLACK)
                    surface.blit(text, (layer_x - text.get_width()//2, neuron_y - text.get_height()//2))
        
        # Draw layer labels
        for layer_idx in range(len(self.activations)):
            layer_x = (layer_idx + 1) * self.layer_spacing
            layer_name = f"Layer {layer_idx+1}"
            text = self.font.render(layer_name, True, COLORS.BLACK)
            surface.blit(text, (layer_x - text.get_width()//2, 20))



visualizer = NeuralNetworkVisualizer(0, 0, 600, 600)



class WindowHandler:
    def __init__(self, windowSurface):
        windowSurface
        
    def ClearScreen(self, windowSurface):
        windowSurface.fill(COLORS.WHITE)

    def RenderStep(self, windowSurface):
        self.ClearScreen(windowSurface)
        visualizer.update_activations()
        visualizer.draw_network(windowSurface)
        pygame.display.update()

    def HandleEvent(event):
        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                # Trigger a training step when space is pressed
                model.train_step()