import neat
import os
import pickle
import pygame
from ai.networks.network import DinoNetwork # Import DinoNetwork class

class AIController:
    def __init__(self, config_file):
        # --- Loading N.E.A.T. configuration ---
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        self.config = config
        
        # --- Creating initial population ---
        self.population = neat.Population(config)
        self.generation = 0
        self.networks = {} # Dictionary to store DinoNetwork objects, keyed by genome_id

        # Initializing with more useful starting values
        self.network_stats = {
            "nodes_created": 0,
            "connections_created": 0,
            "mutations": 0,
            "species_count": 1,  # Starting with at least 1 species
            "best_fitness": 0
        }
        
        # Tracking the initial structure size for proper delta calculation
        self._initial_node_count = sum(len(genome.nodes) for genome in self.population.population.values())
        self._initial_conn_count = sum(len(genome.connections) for genome in self.population.population.values())
        self._previous_best_genome_id = None

    def reset(self):
        """Resets the population to start with fresh genomes."""
        # Creating a new population with the same config
        self.population = neat.Population(self.config)
        self.generation = 0
        self.networks = {}  # Clearing any cached networks

        # Resetting network stats
        self.network_stats = {
            "nodes_created": 0,
            "connections_created": 0,
            "mutations": 0,
            "species_count": 1,
            "best_fitness": 0
        }
        
        # Resetting tracking variables
        self._initial_node_count = sum(len(genome.nodes) for genome in self.population.population.values())
        self._initial_conn_count = sum(len(genome.connections) for genome in self.population.population.values())
        self._previous_best_genome_id = None

    def save_best_genome(self, filename="tests/best_genome.pkl"):
        """Saves the best genome to a file."""
        best_genome = self.get_best_genome()
        if best_genome:
            with open(filename, 'wb') as f:
                pickle.dump(best_genome, f)
            print(f"Best genome saved to {filename}")
        else:
            print("No best genome to save")

    def save_population(self, filename="tests/population.pkl"):
        """Saves the entire population to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.population, f)
        print(f"Population saved to {filename}")

    def load_best_genome(self, filename="tests/best_genome.pkl"):
        """Loads the best genome from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                best_genome = pickle.load(f)
            print(f"Best genome loaded from {filename}")
            return best_genome
        else:
            print(f"File {filename} not found")
            return None

    def load_population(self, filename="tests/population.pkl"):
        """Loads a saved population."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.population = pickle.load(f)
            print(f"Population loaded from {filename}")
            
            # Updating our tracking variables to match loaded population
            self._initial_node_count = sum(len(genome.nodes) for genome in self.population.population.values())
            self._initial_conn_count = sum(len(genome.connections) for genome in self.population.population.values())
            
            # Resetting network cache
            self.networks = {}
            
            return True
        else:
            print(f"File {filename} not found")
            return False
    
    def get_action(self, sensor_data, genome_id):
        """Decides Dino action using the N.E.A.T.-evolved neural network for the given genome.
        Args:
            sensor_data: A dictionary containing sensor data (distance, type_encoded, speed).
            genome_id: The ID of the genome for which to get the action.
        Returns:
            A string representing the action: "jump", "duck", or "run"."""
        if genome_id not in self.networks: # Creating DinoNetwork if not already created for this genome_id
            genome = self.population.population.get(genome_id)
            if genome:  # Making sure the genome exists
                self.networks[genome_id] = DinoNetwork(genome, self.config)
            else:
                print(f"Warning: Genome {genome_id} not found in population")
                return "run"  # Default action if genome not found

        network = self.networks[genome_id] # Getting DinoNetwork for this genome_id
        return network.get_action(sensor_data) # Using DinoNetwork to get action based on sensor data

    def load_best_genome(self, filename="tests/best_genome.pkl"):
        """Loads the best genome from a file."""
        import pickle
        import os
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                best_genome = pickle.load(f)
            print(f"Best genome loaded from {filename}")
            return best_genome
        else:
            print(f"File {filename} not found")
            return None

    def load_population(self, filename="tests/population.pkl"):
        """Loads a saved population."""
        import pickle
        import os
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.population = pickle.load(f)
            print(f"Population loaded from {filename}")
            return True
        else:
            print(f"File {filename} not found")
            return False
    
    def evaluate_genomes(self, genomes, config):
        """Fitness function for N.E.A.T. - evaluates the fitness of each genome.
        For now, placeholder - fitness will be assigned externally in dino_game.py."""
        pass

    def run_generation(self):
        """Runs one generation of the N.E.A.T. evolutionary algorithm."""
        self.generation += 1
        print(f"\n--- Generation {self.generation} ---")

        # Tracking network stats before reproduction
        prev_connections = sum(len(genome.connections) for genome in self.population.population.values())
        prev_nodes = sum(len(genome.nodes) for genome in self.population.population.values())
        prev_species = len(self.population.species.species)

        # --- Reproduction and next generation ---
        self.population.next_generation()
        self.networks = {} # Clearing network cache for new generation (networks will be recreated on demand)

        # Tracking network changes after reproduction
        curr_connections = sum(len(genome.connections) for genome in self.population.population.values())
        curr_nodes = sum(len(genome.nodes) for genome in self.population.population.values())
        curr_species = len(self.population.species.species)
        
        # Updating network stats more effectively
        self.network_stats["nodes_created"] += max(0, curr_nodes - prev_nodes)
        self.network_stats["connections_created"] += max(0, curr_connections - prev_connections)
        self.network_stats["mutations"] += 1  # Increment mutations counter each generation
        self.network_stats["species_count"] = curr_species

        # Finding the best genome and update the best fitness
        best_genome = None
        best_fitness = 0.0
        best_genome_id = None
        
        for genome_id, genome in self.population.population.items():
            if hasattr(genome, 'fitness') and genome.fitness is not None:
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome
                    best_genome_id = genome_id
        
        # Updating best fitness in network stats
        if best_fitness > self.network_stats["best_fitness"]:
            self.network_stats["best_fitness"] = best_fitness
            
        # Tracking if the best genome changed significantly
        if self._previous_best_genome_id != best_genome_id:
            self._previous_best_genome_id = best_genome_id
            # We have a new best genome - this contributes to evolution progress

    def get_best_genome(self):
        """Returns the best genome based on fitness."""
        best_genome = None
        best_fitness = -1
        
        for genome_id, genome in self.population.population.items():
            if hasattr(genome, 'fitness') and genome.fitness is not None:
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome
        
        return best_genome

    def draw_network_structure(self, screen, x, y, width, height):
        """Draw a simplified visualization of the best network's structure."""
        # Finding the best genome
        best_genome = self.get_best_genome()
        
        if not best_genome:
            # If no genome has fitness values yet, create an empty visualization
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            surface.fill((255, 255, 255, 180))  # Light gray background with transparency
            # surface.fill((240, 240, 240, 180))  # Light gray background with transparency
            
            font = pygame.font.SysFont("Consolas", 14, True)
            title = "Network Structure (No data yet)"
            title_surface = font.render(title, True, (0, 0, 0))
            surface.blit(title_surface, (10, 10))
            
            screen.blit(surface, (x - width//2, y - height//2))  # Center the surface
            return
            
        # Creating a surface for drawing
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 180))  # Light gray background with transparency
        # surface.fill((240, 240, 240, 180))  # Light gray background with transparency
        
        # Drawing a border around the visualization
        # pygame.draw.rect(surface, (0, 0, 0), (0, 0, width, height), 1)
        
        # Counting layers (input, hidden, output)
        input_nodes = []
        output_nodes = []
        hidden_nodes = []
        
        # Creating the node lists with proper ordering
        for i in range(self.config.genome_config.num_inputs):
            input_nodes.append(-i - 1)  # NEAT uses negative IDs for input nodes
            
        for i in range(self.config.genome_config.num_outputs):
            output_nodes.append(i)  # NEAT uses consecutive IDs starting at 0 for output nodes
            
        for node_id in best_genome.nodes.keys():
            if node_id not in input_nodes and node_id not in output_nodes:
                hidden_nodes.append(node_id)
        
        # Calculating positions
        node_radius = 6  # Slightly larger node radius
        layer_width = width / 3
        
        # Drawing connections first (so they appear behind nodes)
        for conn_gene in best_genome.connections.values():
            if conn_gene.enabled:
                input_idx = conn_gene.key[0]
                output_idx = conn_gene.key[1]
                
                # Determining source layer and position
                if input_idx in input_nodes:
                    src_layer = 0
                    src_y = height * (0.2 + 0.6 * (input_nodes.index(input_idx) / max(1, len(input_nodes))))
                elif input_idx in hidden_nodes:
                    src_layer = 1
                    src_y = height * (0.2 + 0.6 * (hidden_nodes.index(input_idx) / max(1, len(hidden_nodes))))
                else:
                    # Skipping connections with unknown source nodes
                    continue
                
                # Determining target layer and position
                if output_idx in hidden_nodes:
                    tgt_layer = 1
                    tgt_y = height * (0.2 + 0.6 * (hidden_nodes.index(output_idx) / max(1, len(hidden_nodes))))
                elif output_idx in output_nodes:
                    tgt_layer = 2
                    tgt_y = height * (0.2 + 0.6 * (output_nodes.index(output_idx) / max(1, len(output_nodes))))
                else:
                    # Skipping connections with unknown target nodes
                    continue
                
                src_x = src_layer * layer_width + layer_width/2
                tgt_x = tgt_layer * layer_width + layer_width/2
                
                # Calculating line color based on weight (red for negative, green for positive)
                weight = conn_gene.weight
                if weight < 0:
                    color = (255, 0, 0)  # Red for negative
                else:
                    color = (0, 200, 0)  # Green for positive
                
                # Calculating line width based on weight magnitude
                line_width = max(1, min(4, abs(int(weight))))
                
                # Drawing the connection line on the surface
                pygame.draw.line(surface, color, 
                                (int(src_x), int(src_y)),
                                (int(tgt_x), int(tgt_y)), 
                                line_width)
        
        # Drawing nodes
        # Input nodes (left side)
        for i, node_id in enumerate(input_nodes):
            node_x = layer_width/2
            node_y = height * (0.2 + 0.6 * (i / max(1, len(input_nodes))))
            pygame.draw.circle(surface, (0, 0, 255), (int(node_x), int(node_y)), node_radius)
        
        # Hidden nodes (middle)
        for i, node_id in enumerate(hidden_nodes):
            node_x = layer_width + layer_width/2
            node_y = height * (0.2 + 0.6 * (i / max(1, len(hidden_nodes))))
            pygame.draw.circle(surface, (0, 0, 0), (int(node_x), int(node_y)), node_radius)
        
        # Output nodes (right side)
        for i, node_id in enumerate(output_nodes):
            node_x = 2 * layer_width + layer_width/2
            node_y = height * (0.2 + 0.6 * (i / len(output_nodes)) if len(output_nodes) > 0 else 0.5)
            pygame.draw.circle(surface, (255, 0, 0), (int(node_x), int(node_y)), node_radius)
        
        # Drawing title
        font = pygame.font.SysFont("Consolas", 14, True)  # Slightly larger font
        title = "Best Network Structure"
        title_surface = font.render(title, True, (0, 0, 0))
        surface.blit(title_surface, (15, 10))
        
        # Drawing the fitness value
        fitness_text = f"Fitness: {best_genome.fitness:.2f}"
        fitness_surface = font.render(fitness_text, True, (0, 0, 0))
        surface.blit(fitness_surface, (15, 25))
        
        # Adding labels for the layers
        input_label = font.render("Inputs", True, (0, 0, 150))
        surface.blit(input_label, (layer_width/2 - 25, height - 50))
        
        hidden_label = font.render("Hidden", True, (0, 0, 0))
        surface.blit(hidden_label, (layer_width + layer_width/2 - 30, height - 50))
        
        output_label = font.render("Outputs", True, (150, 0, 0))
        surface.blit(output_label, (2 * layer_width + layer_width/2 - 35, height - 50))
        
        # Drawing the surface to the screen, centered at the provided coordinates
        screen.blit(surface, (x - width//2, y - height//2))