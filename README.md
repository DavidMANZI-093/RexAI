# RexAI - NEAT-Powered Dinosaur Game AI

## 🦖 Project Overview

RexAI is an AI-powered implementation of the Chrome Dinosaur game using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. Watch as neural networks evolve to learn and master the game through generations of training!

This project demonstrates how evolutionary algorithms can be used to create intelligent agents that learn through simulation, without being explicitly programmed with game rules.

## ✨ Features

- 🦖 Chrome Dinosaur-inspired game implementation using PyGame
- 🧠 NEAT algorithm implementation for neural network evolution
- 📊 Real-time visualization of network structure
- 🔄 Save/load functionality for continuing training sessions
- 📈 Training management system with species configuration
- 📝 Command-line interface for controlling training parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyGame
- NEAT-Python
- Colorama

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rexai.git
cd rexai
```

2. Install dependencies:
```bash
pip install pygame neat-python colorama
```

3. Run the game:
```bash
python -m rexai
```

## 🎮 Usage Options

RexAI comes with several command-line options to control the training process:

```bash
python -m rexai --config species.json --species "Rex 7 g-20.0" --generations 200
```

### Available Arguments

- `--config`, `-c`: Path to JSON configuration file
- `--species`, `-s`: Name of the species to load
- `--fresh`, `-f`: Start a new training session (ignores existing population)
- `--list`, `-l`: List all available species in the configuration
- `--generations`, `-g`: Number of generations to run (default: 100)

## 🧬 How It Works

RexAI uses NEAT (NeuroEvolution of Augmenting Topologies), an evolutionary algorithm that evolves neural networks. The system works through the following process:

1. **Initialization**: Creates a population of simple neural networks
2. **Evaluation**: Each network controls a dinosaur in the game
3. **Selection**: Networks that perform better (survive longer) have higher fitness scores
4. **Reproduction**: Top-performing networks reproduce, with mutations introducing variations
5. **Iteration**: The process repeats, evolving more sophisticated behaviors over generations

### Neural Network Inputs

The AI receives several inputs about the game state:
- Distance to the next obstacle
- Type of obstacle (cactus or bird)
- Game speed
- Dinosaur state (jumping, ducking, running)
- Obstacle dimensions and position

### Neural Network Outputs

The network decides between three possible actions:
- Jump
- Duck
- Run (continue running normally)

## 🏗️ Project Structure

```
rexai/
├── __init__.py
├── main.py                   # Main entry point
├── controllers/
│   └── ai_controller.py      # NEAT controller implementation
├── ai/
│   └── networks/
│       └── network.py        # DinoNetwork class (neural network)
├── game/
│   ├── dino_game.py          # Game loop and rendering
│   ├── dino.py               # Dinosaur player entity
│   └── obstacle.py           # Game obstacles (cacti and birds)
├── utils/
│   ├── config_manager.py     # Handles configuration and species
│   └── training_manager.py   # Manages training process
├── config/
│   └── neat_config.txt       # NEAT algorithm parameters
├── tests/                    # Directory for saved populations/genomes
└── data/                     # Pre-trained models
```

## 💾 Training and Saving Progress

RexAI automatically saves progress every 10 generations. You can load a previous training session using the species configuration file.

```bash
# List available species
python -m rexai --config species.json --list

# Load a specific species and continue training
python -m rexai --config species.json --species "Rex 7 g-20.0" --generations 200

# Start fresh training
python -m rexai --fresh --generations 300
```

## 🔧 Configuration

The `neat_config.txt` file contains parameters for the NEAT algorithm, including:
- Population size
- Mutation rates
- Network structure parameters
- Species compatibility thresholds

The `species.json` file manages different trained "species" of dinosaurs, allowing you to save and load different training runs.

## 📝 License

[MIT License](LICENSE)

## 👥 Acknowledgements

- [NEAT-Python](https://neat-python.readthedocs.io/) for the NEAT implementation
- [PyGame](https://www.pygame.org/) for the game engine
- The Chrome Dinosaur game for inspiration
