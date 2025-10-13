# Importing pygame module
import time

import pygame
from pygame.locals import *
from math import log, dist

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
        # Get the appropriate network based on type
        if self.network_type == "rabbit":
            self.net = model.rabbit_net
        else:
            self.net = model.wolf_net
            
        # Get the latest activations from the model
        if hasattr(self.net, 'layer_activations') and len(self.net.layer_activations) > 0:
            self.activations = self.net.layer_activations
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



class PositionVisualizer:
    def __init__(self, startX, startY, width, height):
        self.start_x = startX
        self.start_y = startY
        self.width = width
        self.height = height
        self.position_history = []
        self.max_history = 100
        self.font = pygame.font.SysFont('Arial', 14)


        
    def update_positions(self, rabbit_positions, wolf_positions, noisy_rabbit_positions):
        # Store all positions from the training step as a single entry
        step_positions = list(zip(rabbit_positions, wolf_positions, noisy_rabbit_positions))
        self.position_history.append(step_positions)
        
        # Keep only the most recent training steps
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)


    AddX = 960
    AddY = 540
    AddX2 = 960
    AddY2 = 540

    def draw(self, surface):
        global AddX, AddY, AddX2, AddY2
        if not self.position_history:
            return
            
        # Get the most recent training step's positions
        recent_step = self.position_history[-1]
        
        # Draw connecting lines to show the path
        if len(recent_step) > 1:
            # Draw rabbit path with increasing thickness for newer positions
            for i in range(len(recent_step) - 1):
                r_pos1 = recent_step[i][0]
                r_pos2 = recent_step[i + 1][0]
                # Newer positions have thicker lines
                line_width = max(1, min(5, int(1 + i/5)))
                color = COLORS.GREEN
                pygame.draw.line(surface, color,
                                (int(r_pos1[0] + self.start_x), int(r_pos1[1] + self.start_y)),
                                (int(r_pos2[0] + self.start_x), int(r_pos2[1] + self.start_y)), line_width)

            # Draw wolf path with increasing thickness for newer positions
            for i in range(len(recent_step) - 1):
                w_pos1 = recent_step[i][1]
                w_pos2 = recent_step[i + 1][1]
                # Newer positions have thicker lines
                line_width = max(1, min(5, int(1 + i/5)))
                color = COLORS.RED
                pygame.draw.line(surface, color,
                                (int(w_pos1[0] + self.start_x), int(w_pos1[1] + self.start_y)),
                                (int(w_pos2[0] + self.start_x), int(w_pos2[1] + self.start_y)), line_width)
        
        # Draw all positions with increasing size for newer positions
        for i, (r_pos, w_pos, n_pos) in enumerate(recent_step):
            # Newer positions have larger circles
            size_factor = min(2.0, 1.0 + i/20)  # Gradually increase size
            
            # Draw rabbit position
            pygame.draw.circle(surface, COLORS.GREEN, 
                              (int(r_pos[0] + self.AddX), int(r_pos[1] + self.AddY)),
                              int(5 * size_factor))
            
            # Draw wolf position
            pygame.draw.circle(surface, COLORS.RED, 
                              (int(w_pos[0] + self.AddX2), int(w_pos[1] + self.AddY2)),
                              int(5 * size_factor))
            x_Sum = r_pos[0] + self.AddX
            y_Sum = r_pos[1] + self.AddY

            if x_Sum >= 1400:self.AddX = -r_pos[0]
            if x_Sum < 0:self.AddX = 1400 - r_pos[0]

            if y_Sum >= 700:self.AddY = -r_pos[1]
            if y_Sum < 0:self.AddY = 700 - r_pos[1]

            x_Sum2 = w_pos[0] + self.AddX2
            y_Sum2 = w_pos[1] + self.AddY2

            if x_Sum2 >= 1400:self.AddX2 = -w_pos[0]
            if x_Sum2 < 0:self.AddX2 = 1400 - w_pos[0]

            if y_Sum2 >= 700:self.AddY2 = -w_pos[1]
            if y_Sum2 < 0:self.AddY2 = 700 - w_pos[1]


            

            
            # Add step numbers to newer positions
            if i > len(recent_step) - 10:  # Only label last 10 positions
                step_text = self.font.render(str(i), True, COLORS.BLACK)
                # Position text slightly above the point
                text_pos = (int(r_pos[0] + self.start_x) - step_text.get_width()//2, 
                           int(r_pos[1] + self.start_y) - 15)
                surface.blit(step_text, text_pos)
        
        # Draw starting and ending positions with special markers
        # if recent_step:
        #     # Starting positions
        #     r_start, w_start, n_start = recent_step[0]
        #     pygame.draw.circle(surface, COLORS.GREEN,
        #                       (int(r_start[0] + self.start_x), int(r_start[1] + self.start_y)),
        #                       8, 2)
        #     pygame.draw.circle(surface, COLORS.RED,
        #                       (int(w_start[0] + self.start_x), int(w_start[1] + self.start_y)),
        #                       8, 2)
        #
        #     # Add "Start" label
        #     start_text = self.font.render("Start", True, COLORS.BLACK)
        #     surface.blit(start_text, (int(r_start[0] + self.start_x) - start_text.get_width()//2,
        #                              int(r_start[1] + self.start_y) - 25))
            
            #Ending positions
            r_end, w_end, n_end = recent_step[-1]
            pygame.draw.circle(surface, COLORS.GREEN,
                              (int(r_end[0] + self.start_x), int(r_end[1] + self.start_y)),
                              10)
            pygame.draw.circle(surface, COLORS.RED,
                              (int(w_end[0] + self.start_x), int(w_end[1] + self.start_y)),
                              10)

            # Add "End" label
            end_text = self.font.render("End", True, COLORS.BLACK)
            surface.blit(end_text, (int(r_end[0] + self.start_x) - end_text.get_width()//2,
                                   int(r_end[1] + self.start_y) + 15))

            #Draw distance between final positions
            distance = dist((w_end[0], w_end[1]), (r_end[0], r_end[1]))
            distance_text = self.font.render(f"Final Distance: {distance:.1f}", True, COLORS.BLACK)
            surface.blit(distance_text, (1100, 650))

            #Draw step count
            step_text = self.font.render(f"Steps: {len(recent_step)}", True, COLORS.BLACK)
            surface.blit(step_text, (1100, 600))

# Create visualizers for both networks
rabbit_model_visualizer = NeuralNetworkVisualizer(50, 100, 200, 200, "rabbit")
wolf_model_visualizer = NeuralNetworkVisualizer(300, 100, 200, 200, "wolf")

class WindowHandler:
    def __init__(self, windowSurface):
        self.windowSurface = windowSurface
        self.position_visualizer = PositionVisualizer(960, 540, 900, 150)
        
    def ClearScreen(self, windowSurface):
        windowSurface.fill(COLORS.WHITE)

    def RenderStep(self, windowSurface):
        self.ClearScreen(windowSurface)

        rabbit_model_visualizer.update_activations()
        wolf_model_visualizer.update_activations()
        rabbit_model_visualizer.draw_network(windowSurface)
        wolf_model_visualizer.draw_network(windowSurface)

        # Update and draw position visualizer with all positions from the training step
        if (model.rabbit_positions and model.wolf_positions and 
            model.noisy_rabbit_positions and len(model.rabbit_positions) > 0):
            self.position_visualizer.update_positions(
                model.rabbit_positions,
                model.wolf_positions,
                model.noisy_rabbit_positions
            )
        self.position_visualizer.draw(windowSurface)


        # Display instructions
        font = pygame.font.SysFont('Arial', 16)
        instructions = [
            "Press SPACE to run 100 training steps (gives worse training results)",
            "Press Q to reset models and positions"
        ]


        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, COLORS.BLACK)
            windowSurface.blit(text_surface, (10, 10 + i * 20))


        model.train_step(1)
        pygame.display.update()
        time.sleep(0.2)




    def HandleEvent(self, event):
        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                # Trigger a training step when space is pressed
                model.train_step(100)
            elif event.key == K_q:
                # Reset models and positions when Q is pressed
                model.reset_models()
                # Clear position history
                self.position_visualizer.position_history = []