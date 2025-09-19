# Importing pygame module
import pygame
from pygame.locals import *
from math import log

import Colors as COLORS
import Models as model

pygame.init()

class NeuralNetworkVisualizer:
    def __init__(self, startX, startY, width, height, network_type="rabbit"):
        self.start_x = startX
        self.start_y = startY
        self.drawing_space_width = width
        self.drawing_space_height = height
        self.network_type = network_type
        
        # Get the appropriate network based on type
        if network_type == "rabbit":
            self.net = model.rabbit_net
            self.title = "Rabbit Network"
        else:
            self.net = model.wolf_net
            self.title = "Wolf Network"
            
        # Initialize with empty activations
        self.layer_spacing = width
        self.neuron_radius = 12
        self.activations = []
        self.font = pygame.font.SysFont('Arial', 14)
        self.update_activations()

    # Neuron color method
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
        if hasattr(self.net, 'layer_activations') and len(self.net.layer_activations) > 0:
            self.activations = self.net.layer_activations
            self.layer_spacing = self.drawing_space_width // (len(self.activations) + 2)
        else:
            # Default activations if none available (now includes input layer)
            self.activations = [[0] * 5, [0] * 10, [0] * 10, [0] * 10, [0] * 6, [0] * 2]
            self.layer_spacing = self.drawing_space_width // (len(self.activations) + 2)

    def draw_neuron_connections(self, surface):
        if not self.activations or len(self.activations) < 2:
            return
            
        # Start from layer 0 (input) to layer 1 (first hidden)
        for layer_idx in range(len(self.activations) - 1):
            current_layer = self.activations[layer_idx]
            next_layer = self.activations[layer_idx + 1]
            
            current_x = (layer_idx + 1) * self.layer_spacing + self.start_x
            next_x = (layer_idx + 2) * self.layer_spacing + self.start_x
            
            for i, current_activation in enumerate(current_layer):
                current_y = self.drawing_space_height * (i + 1) / (len(current_layer) + 1) + self.start_y
                
                for j, next_activation in enumerate(next_layer):
                    next_y = self.drawing_space_height * (j + 1) / (len(next_layer) + 1) + self.start_y
                    
                    # Calculate line properties based on activations
                    alpha = min(255, int(abs(current_activation) * 255))
                    width = int(log(max(1, int(abs(current_activation) * 3))))
                    
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
        if not self.activations:
            return
            
        # Draw connections first (behind neurons)
        self.draw_neuron_connections(surface)
        
        # Draw neurons
        for layer_idx, layer_activations in enumerate(self.activations):
            neuron_count = len(layer_activations)
            layer_x = (layer_idx + 1) * self.layer_spacing + self.start_x
            
            for neuron_idx in range(neuron_count):
                neuron_y = self.drawing_space_height * (neuron_idx + 1) / (neuron_count + 1) + self.start_y
                # Get activation value and convert to color
                activation = layer_activations[neuron_idx]
                color = self.GetNeuronColor(activation)
                
                # Draw neuron
                pygame.draw.circle(surface, color, (layer_x, neuron_y), self.neuron_radius)
                pygame.draw.circle(surface, COLORS.BLACK, (layer_x, neuron_y), self.neuron_radius, 1)
                
                # Draw activation value
                if abs(activation) > 0.01:  # Only show significant activations
                    text = self.font.render(f"{activation:.2f}", True, COLORS.BLACK)
                    surface.blit(text, (layer_x - text.get_width()//2, neuron_y - text.get_height()//2))
        
        # Draw layer labels with special names for input and output layers
        for layer_idx in range(len(self.activations)):
            layer_x = (layer_idx + 1) * self.layer_spacing + self.start_x
            
            if layer_idx == 0:
                layer_name = "Input"
            elif layer_idx == len(self.activations) - 1:
                layer_name = "Output"
            else:
                layer_name = f"Hidden {layer_idx}"
                
            text = self.font.render(layer_name, True, COLORS.BLACK)
            surface.blit(text, ((layer_x - text.get_width()) * 1.05, self.start_y))
        
        # Draw network title
        title_text = self.font.render(self.title, True, COLORS.BLACK)
        surface.blit(title_text, (self.start_x + self.drawing_space_width//2 - title_text.get_width()//2, self.start_y - 30))


# Create visualizers for both networks
rabbit_visualizer = NeuralNetworkVisualizer(50, 100, 450, 400, "rabbit")
wolf_visualizer = NeuralNetworkVisualizer(500, 100, 450, 400, "wolf")

class WindowHandler:
    def __init__(self, windowSurface):
        self.windowSurface = windowSurface
        
    def ClearScreen(self, windowSurface):
        windowSurface.fill(COLORS.WHITE)

    def RenderStep(self, windowSurface):
        self.ClearScreen(windowSurface)
        rabbit_visualizer.update_activations()
        wolf_visualizer.update_activations()
        rabbit_visualizer.draw_network(windowSurface)
        wolf_visualizer.draw_network(windowSurface)
        pygame.display.update()

    def HandleEvent(self, event):
        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                # Trigger a training step when space is pressed
                model.train_step()