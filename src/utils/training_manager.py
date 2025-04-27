import os
import time
import colorama
from colorama import Fore

# Importing configuration manager
from .config_manager import ConfigManager

# Initializing colorama
colorama.init(autoreset=True)

class TrainingManager:
    """
    Manages the training process for the Dino N.E.A.T. AI.
    Integrates the configuration management with the AI controller and game logic.
    """
    def __init__(self):
        """Initializing the training manager"""
        self.config_manager = ConfigManager()
        self.generations = self.config_manager.get_generations()
        self.species_name = self.config_manager.get_species_name()
        self.start_fresh = self.config_manager.should_start_fresh()
        self.genome_path, self.population_path = self.config_manager.get_training_paths()
        
    def setup_training(self, ai_controller):
        """
        Setting up the training environment based on configuration
        - **ai_controller**: The AIController instance to configure
        """
        if self.start_fresh:
            print(f"  - Starting fresh training session...")
            ai_controller.reset()  # Resetting to create a fresh population
            return

        # If we have a population file, trying to load it
        if self.population_path and os.path.exists(self.population_path):
            print(f"  - Loading population from: {Fore.CYAN}{self.population_path}")
            success = ai_controller.load_population(self.population_path)
            
            if success:
                print(f"  - Population loaded successfully!")
                print(f"  - Continuing training with species: {Fore.CYAN}'{self.species_name}'")
            else:
                print(f"  - {Fore.YELLOW}Warning: {Fore.RESET}Failed to load population. Starting fresh.")
                ai_controller.reset()
        # If we only have a best genome, trying to load that
        elif self.best_genome_path and os.path.exists(self.best_genome_path):
            print(f"  - {Fore.YELLOW}Warning: {Fore.RESET}Population file not found or not specified.")
            print(f"  - Loading best genome from: {Fore.CYAN}'{self.best_genome_path}'")
            
            best_genome = ai_controller.load_best_genome(self.best_genome_path)
            if best_genome:
                print(f"  - Best genome loaded successfully!")
                print(f"  - {Fore.MAGENTA}Note:{Fore.RESET} Starting fresh population with this genome as a seed would require code modification.")
                print(f"  - Starting with fresh population instead.")
                ai_controller.reset()
            else:
                print(f"  - {Fore.YELLOW}Warning:{Fore.RESET} Failed to load best genome. Starting fresh.")
                ai_controller.reset()
        else:
            print(f"  - {Fore.YELLOW}Warning:{Fore.RESET} No valid population or genome files specified. Starting fresh.")
            ai_controller.reset()
    
    def get_checkpoint_prefix(self) -> str:
        """
        Get a prefix for checkpoint filenames based on species name
        - **str**: A prefix for checkpoint filenames
        """
        if self.species_name:
            # Converting spaces to underscores and make lowercase
            return self.species_name.lower().replace(" ", "_")
        else:
            # Using a timestamp if no species name is available
            timestamp = time.strftime("%Y%m%d-%H%M")
            return f"{timestamp}"
            
    def save_progress(self, ai_controller, generation: int) -> None:
        """
        Save training progress (population and best genome)
        - **ai_controller**: The AIController instance
        - **generation**: The current generation number
        """
        prefix = self.get_checkpoint_prefix()
        
        # Creating checkpoints directory if it doesn't exist
        os.makedirs("tests", exist_ok=True)
        
        # Saving the population
        population_path = f"tests/test_{prefix}_population.pkl"
        ai_controller.save_population(population_path)
        print(f"  - Population saved to: {Fore.CYAN}'{population_path}'")
        
        # Saving the best genome
        best_genome_path = f"tests/test_{prefix}_genome.pkl"
        ai_controller.save_best_genome(best_genome_path)
        print(f"  - Best genome saved to: {Fore.CYAN}'{best_genome_path}'")
        
        # Also saving periodic checkpoints
        if generation % 10 == 0:  # Save every 10 generations
            checkpoint_path = f"tests/test_{prefix}_last-gen-count_{generation}.pkl"
            ai_controller.save_population(checkpoint_path)
            print(f"  - Checkpoint saved to: {Fore.CYAN}'{checkpoint_path}'")