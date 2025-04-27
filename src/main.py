import os

# Local imports
from .controllers.ai_controller import AIController
from .game.dino_game import main as run_dino_game
from .utils.training_manager import TrainingManager

def main():
    """Main entry point for the program"""
    # Creating training manager (which also parses command line arguments)
    training_manager = TrainingManager()
    
    # Initializing the AI controller with the NEAT config
    config_path = os.path.join("config", "neat_config.txt")  # Path to existing NEAT config
    ai_controller = AIController(config_path)
    
    # Setting up training based on config and command line arguments
    training_manager.setup_training(ai_controller)
    
    # Running the game with the configured AI controller
    try:
        # Running the Dino game with configured AI controller
        run_dino_game(ai_controller, training_manager)
    except KeyboardInterrupt:
        print("\n  - Training interrupted by user")
    finally:
        # Always saving progress when finishing or interrupting
        training_manager.save_progress(ai_controller, ai_controller.generation)
        
    print("  - Training complete!")

if __name__ == "__main__":
    main()