import neat

class DinoNetwork:
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config) # Creating FeedForward Network from genome and config

    def get_outputs(self, inputs):
        """Passes inputs through the neural network and return the outputs.
        Args:
            inputs: A list or tuple of sensor inputs (distance, type, speed).
        Returns:
            A list of neural network outputs (for jump, duck, run actions)."""
        return self.net.activate(inputs) # Using neat-python's network activation
    
    def get_action(self, sensor_data):
        """Decides Dino action based on neural network outputs.
        Args:
            sensor_data: A dictionary containing sensor data (distance, type, speed).
        Returns:
            A string representing the action: "jump", "duck", or "run"."""
        # Preparing inputs - ensuring they're all single numeric values
        distance = float(sensor_data["distance"])
        type_encoded = float(sensor_data["type_encoded"])
        speed = float(sensor_data["speed"])
        obstacle_width = float(sensor_data["obstacle_width"])
        obstacle_height = float(sensor_data["obstacle_height"])
        obstacle_x = float(sensor_data["obstacle_x"])
        obstacle_y = float(sensor_data["obstacle_y"])
        dino_x = float(sensor_data["dino_x"])
        dino_y = float(sensor_data["dino_y"])
        dino_width = float(sensor_data["dino_width"])
        dino_height = float(sensor_data["dino_height"])
        dino_state = float(sensor_data["dino_state"])

        # inputs = (distance, type_encoded, speed)  # Tuple of 3 inputs matching config
        inputs = (distance, type_encoded, speed, obstacle_width, obstacle_height, obstacle_x, obstacle_y, dino_x, dino_y, dino_width, dino_height, dino_state)  # Tuple of 12 inputs matching config
        
        try:
            output = self.get_outputs(inputs) # Getting neural network outputs
            
            jump_output = output[0] # Output neuron 0 for jump action
            duck_output = output[1] # Output neuron 1 for duck action
            run_output = output[2] # Output neuron 2 for run action
            
            # Debug output
            # print(f"Network outputs: Jump={jump_output:.2f}, Duck={duck_output:.2f}, Run={run_output:.2f}")
            
            # --- Action decision logic based on outputs ---
            if jump_output > 0.5: # Threshold for jump action
                return "jump"
            elif duck_output > 0.5: # Threshold for duck action
                return "duck"
            elif run_output > 0.5: # Threshold for run action
                return "run" # Default action
        except Exception as e:
            print(f"Error in neural network: {e}")
            print(f"Inputs were: {inputs}")
            return "run"  # Default to run on error