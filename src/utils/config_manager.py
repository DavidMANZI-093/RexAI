import os
import json
import argparse
import colorama
from colorama import Fore, Style

# Initializing colorama for cross-platform colored terminal output
colorama.init(autoreset=True)

class ConfigManager:
    """
    Manages configuration for the Dino NEAT AI project.
    Loads species configurations from JSON and provides interfaces for selecting
    and using saved genomes and populations.
    """
    def __init__(self):
        """Initializing the configuration manager"""
        self.config_path = None
        self.species_config = {"species": []}
        self.selected_species = None
        self.start_fresh = False
        self._parse_arguments()
        
    def _parse_arguments(self):
        """Parsing command line arguments"""
        parser = argparse.ArgumentParser(
            description='RexAI Training Program',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to JSON configuration file'
        )
        
        parser.add_argument(
            '--species', '-s',
            type=str,
            help='Name of the species to load'
        )
        
        parser.add_argument(
            '--fresh', '-f',
            action='store_true',
            help='Start a new training session (ignores existing population)'
        )
        
        parser.add_argument(
            '--list', '-l',
            action='store_true',
            help='List all available species in the configuration'
        )

        parser.add_argument(
            '--generations', '-g',
            type=int,
            default=100,
            help='Generation number to start from (default: 100)'
        )
        
        args = parser.parse_args()
        
        # Storing the arguments
        self.config_path = args.config
        self.species_name = args.species
        self.start_fresh = args.fresh
        self.list_species = args.list
        self.generations = args.generations
        
        # If a config file was provided, loading it
        if self.config_path:
            self._load_config()
            
            # If --list flag is set, printing species and exit
            if self.list_species:
                self.print_species_list()
                exit(0)
                
            # Selecting species based on command line args or user input
            self._select_species()
        else:
            print(f"  - {Fore.YELLOW}Warning:{Fore.RESET} No configuration file specified. Starting with default settings.")
            self.start_fresh = True
            
    def _load_config(self):
        """Loading the configuration from the specified JSON file"""
        try:
            if not os.path.exists(self.config_path):
                print(f"  - {Fore.RED}Error:{Fore.RESET} Configuration file {Fore.CYAN}'{self.config_path}'{Fore.RESET} not found.")
                exit(1)
                
            with open(self.config_path, 'r') as f:
                self.species_config = json.load(f)
                
            # Validate basic structure
            if "species" not in self.species_config or not isinstance(self.species_config["species"], list):
                print(f"  - {Fore.RED}Error:{Fore.RESET} Invalid configuration format. Missing {Fore.YELLOW}'species'{Fore.RESET} list.")
                exit(1)
                
            print(f"\n  - Configuration loaded successfully from {Fore.CYAN}'{self.config_path}'")
            print(f"  - Found {Fore.MAGENTA}{len(self.species_config['species'])}{Fore.RESET} species configurations")
            
        except json.JSONDecodeError:
            print(f"   - {Fore.RED}Error:{Fore.RESET} Configuration file {Fore.CYAN}'{self.config_path}'{Fore.RESET} is not valid JSON.")
            exit(1)
        except Exception as e:
            print(f"  - {Fore.RED}Error loading configuration: {str(e)}")
            exit(1)
            
    def _select_species(self):
        """
        Selecting a species based on command line args or user input
        """
        # If no species name provided and not starting fresh, prompt user
        if not self.species_name and not self.start_fresh:
            self.print_species_list()
            if self.species_config["species"].__len__() > 0:
                while True:
                    selection = input(f"  > Select a species by number or name (or {Fore.CYAN}'new'{Fore.RESET} to start fresh): {Style.RESET_ALL}")
                    
                    # Check if user wants to start fresh
                    if selection.lower() == 'new':
                        self.start_fresh = True
                        print(f"  - Starting fresh training session with no pre-trained model.")
                        return
                        
                    # Trying to interpret as index
                    try:
                        if selection.isdigit():
                            idx = int(selection) - 1
                            if 0 <= idx < len(self.species_config["species"]):
                                self.selected_species = self.species_config["species"][idx]
                                break
                            else:
                                print(f"    - {Fore.RED}Error: {Fore.RESET}Invalid selection. Please enter a number between 1 and {len(self.species_config['species'])}")
                        else:
                            # Trying to find by name
                            for species in self.species_config["species"]:
                                if species["name"].lower() == selection.lower():
                                    self.selected_species = species
                                    break
                            
                            if self.selected_species:
                                break
                            else:
                                print(f"    - {Fore.RED}Error:{Fore.RESET} Species {Fore.CYAN}'{selection}'{Fore.RESET} not found in configuration.")
                    except Exception as e:
                        print(f"  - {Fore.RED}Error: {Fore.RESET}{str(e)}")
            else:
                exit(1)
        
        # If species name was provided, finding it in the config
        elif self.species_name and not self.start_fresh:
            for species in self.species_config["species"]:
                if species["name"].lower() == self.species_name.lower():
                    self.selected_species = species
                    break
                    
            if not self.selected_species:
                print(f"  - {Fore.RED}Error:{Fore.RESET} Species {Fore.CYAN}'{self.species_name}'{Fore.RESET} not found in configuration.")
                self.print_species_list()
                exit(1)
        
        # If a species was selected, print confirmation
        if self.selected_species:
            print(f"{Fore.GREEN}Selected species: {self.selected_species['name']}")
            
            # Validate paths exist
            self._validate_species_paths()
    
    def _validate_species_paths(self):
        """Validate that the paths in the selected species configuration exist"""
        if not self.selected_species:
            return
            
        # Check best genome path
        if "best_genome_path" in self.selected_species:
            path = self.selected_species["best_genome_path"]
            if not os.path.exists(path):
                print(f"  - {Fore.YELLOW}Warning:{Fore.RESET} Best genome file {Fore.CYAN}'{path}'{Fore.RESET} not found.")
        else:
            print(f"  - {Fore.YELLOW}Warning:{Fore.RESET} No best genome path specified for this species.")
            
        # Check population path
        if "best_population_path" in self.selected_species:
            path = self.selected_species["best_population_path"]
            if not os.path.exists(path):
                print(f"  - {Fore.YELLOW}Warning:{Fore.RESET} Population file {Fore.CYAN}'{path}'{Fore.RESET} not found.")
        else:
            print(f"  - {Fore.YELLOW}Warning:{Fore.RESET} No population path specified for this species.")
            
    def print_species_list(self) -> None:
        """Printing a list of all available species in the configuration"""
        if not self.species_config["species"]:
            print(f"  - {Fore.YELLOW}Warning: {Fore.RESET}No species found in configuration.")
            return
            
        print(f"\n  - Available Species:")
        
        for i, species in enumerate(self.species_config["species"]):
            print(f"    {i+1}. {Fore.BLUE}{species['name']}")
            
        print()
        
    def get_training_paths(self):
        """
        Getting the paths for the best genome and population based on the selected configuration
        """
        genome_path = None
        population_path = None
        
        if self.selected_species and not self.start_fresh:
            genome_path = self.selected_species.get("best_genome_path")
            population_path = self.selected_species.get("best_population_path")
            
        return genome_path, population_path
        
    def get_species_name(self):
        """
        Getting the name of the selected species
        - **str**: The name of the selected species, or None if no species is selected
        """
        if self.selected_species:
            return self.selected_species.get("name")
        return None

    def get_generations(self):
        """
        Getting the number of generations to run
        - **int**: The number of generations to run
        """
        return self.generations
        
    def should_start_fresh(self):
        """
        Checking if a fresh training session should be started
        - **bool**: True if a fresh session should be started, False otherwise
        """
        return self.start_fresh