import pygame
import random
import os
from .dino import Dino
from .obstacle import Obstacle
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

def main(ai_controller=None, training_manager=None):
    pygame.init()

    # --- Screen dimensions ---
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Update window title to include species name if available
    title = "RexAI - Evolving Rexs"
    if training_manager and training_manager.species_name:
        title = f"RexAI - Training {training_manager.species_name}"
    pygame.display.set_caption(title)

    # --- Ground parameters ---
    ground_y = 500
    ground_color = (0, 0, 0)
    ground_thickness = 2
    ground_padding_left = 20
    ground_padding_right = 20

    # --- Game clock ---
    clock = pygame.time.Clock()
    fps = 60

    # If no AI controller is provided, create one with default settings
    if not ai_controller:
        config_path = os.path.join("config", "neat_config.txt")
        from ..controllers.ai_controller import AIController
        ai_controller = AIController(config_path)
        
        # Load saved population if available
        population_file = "tests/saved_population.pkl"
        if os.path.exists(population_file):
            print(f"{Fore.CYAN}Loading population from {population_file}")
            ai_controller.load_population(population_file)
            print(f"{Fore.GREEN}Population loaded successfully!")
        else:
            print(f"{Fore.YELLOW}No saved population found. Starting with fresh population.")

    # --- Game Loop Variables ---
    generation_count = ai_controller.generation

    def run_dino_game_generation(genomes, config):
        """
        Function to run a single generation of the Dino game.
        This function initializes the game, spawns obstacles, and evaluates the fitness of each genome in the population.
        It also handles the game loop, including user input, obstacle management, and rendering.
        - **genomes**: List of genomes to evaluate.
        - **config**: Configuration object for N.E.A.T.
        """
        nonlocal generation_count
        generation_count += 1
        print(f"  - Running Generation {Fore.MAGENTA}{generation_count}{Fore.RESET}...")
        
        dinos = {}  # Dictionary to store Dino objects, keyed by genome_id
        nets = {}  # Dictionary to store neural networks, keyed by genome_id
        
        generation_start_time = pygame.time.get_ticks()  # Start time for generation
        
        for genome_id, genome in genomes:  # Creating Dino objects for each genome in the population
            dinos[genome_id] = Dino()
            # Storing the actual genome object in the ai_controller
            ai_controller.population.population[genome_id] = genome

        # --- Obstacle Management ---
        obstacle_group = pygame.sprite.Group()  # Creating sprite group to hold obstacles
        obstacle_spawn_timer = 0  # Timer to control obstacle spawning frequency
        obstacle_spawn_interval = 60  # Spawning new obstacle every 60 frames (Adjustable frequency)

        game_speed_base = 10  # Base game speed
        game_speed_increment_interval = 7500  # Increasing speed every 500 frames
        game_speed_increment_Amount = 0.5  # Amount to increase speed by
        current_game_speed = game_speed_base  # Starting with base speed
        last_speed_increment_time = pygame.time.get_ticks()  # Tracking last speed increment time

        running_generation = True
        while running_generation:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()  # Quitting game properly
                    quit()  # Quitting Python program

            remaining_dinos = 0
            for genome_id, dino in list(dinos.items()):  # Updating and controlling each Dino in population
                if not dino.dead:  # Only updating living Dinos
                    remaining_dinos += 1
                    dino.update()

                    # --- Sensor Data Extraction for each Dino ---
                    next_obstacle_distance = float('inf')
                    next_obstacle_type_encoded = 0  # Single integer encoding (default 0)
                    closest_obstacle = None

                    for obstacle in obstacle_group:
                        if obstacle.rect.x > dino.rect.right:
                            distance = obstacle.rect.left - dino.rect.right  # Distance to leading edge
                            if distance < next_obstacle_distance:
                                next_obstacle_distance = distance
                                closest_obstacle = obstacle

                    # Normalizing distance for neural network input
                    normalized_distance = min(1.0, next_obstacle_distance / SCREEN_WIDTH)

                    if closest_obstacle:  # If a closest obstacle is found, encode its type
                        obstacle_type = closest_obstacle.type
                        if obstacle_type == "cactus_small":
                            next_obstacle_type_encoded = 0.2  # Using normalized values
                        elif obstacle_type == "cactus_large":
                            next_obstacle_type_encoded = 0.4
                        elif obstacle_type == "cactus_mixed":
                            next_obstacle_type_encoded = 0.6
                        elif obstacle_type == "bird":
                            next_obstacle_type_encoded = 0.8
                        else:
                            next_obstacle_type_encoded = 0.0

                    # Normalizing game speed
                    normalized_game_speed = current_game_speed / 25.0  # Assuming max speed around 20

                    sensor_data = {  # Sensor data for each Dino
                        "distance": normalized_distance,
                        "type_encoded": next_obstacle_type_encoded,
                        "speed": normalized_game_speed,
                        "obstacle_width": closest_obstacle.rect.width if closest_obstacle else 0,
                        "obstacle_height": closest_obstacle.rect.height if closest_obstacle else 0,
                        "obstacle_x": closest_obstacle.rect.x if closest_obstacle else 0,
                        "obstacle_y": closest_obstacle.rect.y if closest_obstacle else 0,
                        "dino_x": dino.rect.x,
                        "dino_y": dino.rect.y,
                        "dino_width": dino.rect.width,
                        "dino_height": dino.rect.height,
                        "dino_state": 0 if dino.is_jumping else (1 if dino.is_ducking else 2)  # 0: Jumping, 1: Ducking, 2: Running
                    }
                    
                    action = ai_controller.get_action(sensor_data, genome_id)  # Getting action for THIS Dino's genome

                    # --- Performing Dino Action based on AI decision ---
                    if action == "jump":
                        if not dino.is_jumping and not dino.is_ducking:  # Preventing jump if already jumping
                            dino.is_jumping = True
                    elif action == "duck" and not dino.is_jumping:  # 'duck' action means ducking (if not jumping)
                        if not dino.is_ducking:  # Preventing duck if already ducking
                            dino.is_ducking = True
                    elif action == "run":  # 'run' action means stop ducking (if ducking)
                        dino.is_ducking = False

            if remaining_dinos == 0:  # If all Dinos are dead, end generation
                running_generation = False  # Ending generation loop
                break

            # --- Obstacle Spawning ---
            obstacle_spawn_timer += 1
            if obstacle_spawn_timer >= obstacle_spawn_interval:
                obstacle_spawn_timer = 0
                obstacle_type_choice = random.choice(["bird", "cactus_mixed", "bird", "cactus_large", "bird", "cactus_small", "bird"])
                new_obstacle = Obstacle(obstacle_type_choice)  # Creating new obstacle
                new_obstacle.speed_x = -current_game_speed  # Setting speed of new obstacle
                obstacle_group.add(new_obstacle)

            # --- Increasing Game Speed over time ---
            if pygame.time.get_ticks() - last_speed_increment_time >= game_speed_increment_interval and current_game_speed <= 25:
                current_game_speed += game_speed_increment_Amount
                last_speed_increment_time = pygame.time.get_ticks()
                for obstacle in obstacle_group:
                    obstacle.speed_x = -current_game_speed

            obstacle_group.update()  # Updating all obstacles in the group

            # --- Collision Detection and Fitness Evaluation ---
            for genome_id, dino in list(dinos.items()):
                if not dino.dead:  # Only checking collision for living Dinos
                    # Make dino a temporary sprite for collision detection
                    temp_sprite = pygame.sprite.Sprite()
                    temp_sprite.rect = dino.rect
                    
                    collision = False
                    for obstacle in obstacle_group:
                        if pygame.sprite.collide_rect(temp_sprite, obstacle):
                            collision = True
                            break
                    
                    if collision:
                        dino.dead = True  # Marking Dino as dead
                        dino.death_time = pygame.time.get_ticks()  # Storing death time for fitness calculation
                        
                        # Finding the genome in genomes
                        for i, (gid, genome) in enumerate(genomes):
                            if gid == genome_id:
                                genomes[i][1].fitness = dino.death_time / 1000.0  # Fitness: survival time in seconds
                                break

            # --- Drawing/Rendering ---
            screen.fill((255, 255, 255))  # Filling screen with white color

            # --- Drawing ground ---
            pygame.draw.line(screen, ground_color,
                             (ground_padding_left, ground_y),
                             (SCREEN_WIDTH - ground_padding_right, ground_y),
                             ground_thickness)
            
            # --- Drawing obstacles ---
            for obstacle in obstacle_group:
                obstacle.draw(screen)  # Drawing each obstacle in the group

            # --- Drawing Dinos ---
            for genome_id, dino in dinos.items():  # Drawing each Dino in the population
                if not dino.dead:
                    dino.draw(screen)

            # Set up font for all text displays
            font = pygame.font.SysFont("Consolas", 16, True)

            # Displaying generation info in center top (keep your existing code)
            generation_text = f"Generation: {generation_count}"
            if training_manager and training_manager.species_name:
                generation_text = f"{training_manager.species_name} - Gen: {generation_count}"
                
            text_surface = font.render(generation_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(20, 20))
            screen.blit(text_surface, text_rect)

            # Displaying remaining dinos and distance in top-left corner
            remaining_text = f"Dinos: {remaining_dinos}"
            generation_current_time = pygame.time.get_ticks() - generation_start_time  # Time since generation started
            distance_text = f"Distance: {int(generation_current_time / 1000)}"  # Simple distance metric based on time

            # Rendering remaining dinos count
            remaining_surface = font.render(remaining_text, True, (0, 0, 0))
            remaining_rect = remaining_surface.get_rect(topleft=(20, 40))
            screen.blit(remaining_surface, remaining_rect)

            # Rendering distance traveled
            distance_surface = font.render(distance_text, True, (0, 0, 0))
            distance_rect = distance_surface.get_rect(topleft=(20, 60))
            screen.blit(distance_surface, distance_rect)

            # Rendering game speed
            speed_text = f"Speed: {int(current_game_speed)}"
            speed_surface = font.render(speed_text, True, (0, 0, 0))
            speed_rect = speed_surface.get_rect(topleft=(20, 80))
            screen.blit(speed_surface, speed_rect)

            ai_controller.draw_network_structure(screen, 650, 150, 280, 280)
            
            pygame.display.flip()  # Updating the display
            clock.tick(fps)
            
        print(f"  - Generation {Fore.MAGENTA}{generation_count}{Fore.RESET} complete. Best fitness: {Fore.MAGENTA}{max([g.fitness for _, g in genomes]):.2f}")

        # Periodic save checkpoints
        if training_manager and generation_count % 10 == 0:  # Save every 10 generations
            training_manager.save_progress(ai_controller, generation_count)
        elif generation_count % 10 == 0:  # Default saving behavior if no training manager
            print(f"  - Creating checkpoint at generation {Fore.MAGENTA}{generation_count}")
            ai_controller.save_population(f"tests/checkpoint_gen_{generation_count}.pkl")

        return  # Generation loop finished
    
    # --- Main N.E.A.T. Evolution loop ---    
    def eval_genomes(genomes, config):
        """
        Function to evaluate genomes in the population.
        This function is called by the N.E.A.T. library to evaluate the fitness of each genome.
        It runs the Dino game for each genome, allowing the AI to control the Dino and evaluate its performance.
        - **genomes**: List of genomes to evaluate.
        - **config**: Configuration object for N.E.A.T.
        """
        run_dino_game_generation(genomes, config)  # Calling game loop function to evaluate genomes

    # --- Run N.E.A.T. Evolution ---
    try:
        if training_manager.generations > 0:
            max_generations = training_manager.generations
            print(f"  - Starting evolution with {Fore.MAGENTA}{max_generations}{Fore.RESET} generations")
        else:
            max_generations = 100 # Default to 100 generations if not specified
            print(f"  - {Fore.YELLOW}Warning: {Fore.RESET}Starting evolution with default {Fore.MAGENTA}{max_generations}{Fore.RESET} generations")
        ai_controller.population.run(eval_genomes, n=max_generations)  # Running evolution for up to 1000 generations
    except KeyboardInterrupt:
        print(f"  - Evolution interrupted by user")
    finally:
        # Saving only happens in the main script now to avoid duplication
        if not training_manager:  # Only save here if not using training_manager
            # Default saving behavior
            print(f"  - Saving final population...")
            ai_controller.save_population("tests/saved_population.pkl")
            print(f"  - Population saved!")

            # Saving the best genome separately
            best_genome = ai_controller.get_best_genome()
            if best_genome:
                print(f"  - Saving best genome...")
                ai_controller.save_best_genome("tests/best_genome.pkl")
                print(f"  - Best genome saved!")

    # Displaying the final best network structure before quitting
    print(f"  - Displaying best network structure...")
    screen.fill((255, 255, 255))
    ai_controller.draw_network_structure(screen, 400, 300, 600, 500)  # Center of screen, larger visualization
    
    # Adding a title
    font = pygame.font.SysFont("Consolas", 24, True)
    title_text = "Final Best Network Structure"
    if training_manager and training_manager.species_name:
        title_text = f"Final Best Network - {training_manager.species_name}"
    title = font.render(title_text, True, (0, 0, 0))
    screen.blit(title, (SCREEN_WIDTH // 2 - 160, 20))
    
    pygame.display.flip()
    
    # Waiting for user to close the window
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
                
    pygame.quit()

if __name__ == "__main__":
    main()