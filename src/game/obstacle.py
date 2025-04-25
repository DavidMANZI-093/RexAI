import pygame
import random
import os  # For file path manipulation

class Obstacle(pygame.sprite.Sprite):
    """
    A class representing an obstacle in the game.
    This class handles the loading of obstacle images, their movement, and animation (if applicable).
    """
    def __init__(self, obstacle_type):
        """
        Initializes the Obstacle class, loading images and setting initial attributes.
        - **obstacle_type**: Type of obstacle (e.g., "cactus_small", "cactus_large", "bird"...).
        """
        super().__init__()

        # Getting the base directory to properly resolve asset paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, "..", "assets", "obstacles")
        
        # Initializing with placeholder in case loading fails
        self.images = []
        self.type = obstacle_type  # Storing the type of obstacle
        
        try:
            if obstacle_type == "cactus_small":
                image_path = os.path.join(assets_dir, "cactus_xs_0.png")
                self.images = [pygame.image.load(image_path)]
            elif obstacle_type == "cactus_large":
                image_path = os.path.join(assets_dir, "cactus_xl_0.png")
                self.images = [pygame.image.load(image_path)]
            elif obstacle_type == "cactus_mixed":
                image_path = os.path.join(assets_dir, "cactus_xx_0.png")
                self.images = [pygame.image.load(image_path)]
            elif obstacle_type == "bird":
                self.images = [
                    pygame.image.load(os.path.join(assets_dir, "bird_0.png")),
                    pygame.image.load(os.path.join(assets_dir, "bird_1.png"))
                ]
                self.animation_speed = 15
                self.animation_counter = 0
                self.current_sprite_index = 0
            else:  # Default to small cactus if type is unrecognized
                image_path = os.path.join(assets_dir, "cactus_xs_0.png")
                self.images = [pygame.image.load(image_path)]
                
            # Scaling all loaded images
            for i in range(len(self.images)):
                self.images[i] = pygame.transform.scale(
                    self.images[i], 
                    (self.images[i].get_width() // 2, self.images[i].get_height() // 2)
                )
                
            # If images were loaded successfully, setting the first one as current
            if self.images:
                self.image = self.images[0]
                self.rect = self.image.get_rect()
            else:
                # Fallback: creating a colored rectangle if images couldn't be loaded
                self.image = pygame.Surface((30, 60))
                self.image.fill((255, 0, 0))  # Red rectangle as fallback
                self.rect = self.image.get_rect()
                print(f"WARNING: No images loaded for obstacle type: {obstacle_type}")
                
        except Exception as e:
            # Fallback if image loading fails
            print(f"Error loading obstacle images: {e}")
            self.image = pygame.Surface((30, 60))
            self.image.fill((255, 0, 0))  # Red rectangle as fallback
            self.rect = self.image.get_rect()

        # Setting position
        self.rect.x = 800  # Initial x position (off-screen to the right)
        self.rect.bottom = 500
        if self.type == "bird":
            self.rect.y = random.choice([410, 385, 355])  # Adjusting y position for bird obstacle

        self.speed_x = -5  # Speed of obstacle movement (leftward)

    def update(self):
        """Moves the obstacle to the left. Removes obstacle if off-screen."""
        self.rect.x += self.speed_x

        if self.rect.right < 0:  # If obstacle is completely off-screen to the left
            self.kill()  # Removing obstacle sprite from all groups

        if self.type == "bird" and hasattr(self, 'animation_counter') and len(self.images) > 1:
            # --- Bird Animation ---
            self.animation_counter += 1
            if self.animation_counter >= self.animation_speed:
                self.animation_counter = 0  # Resetting counter
                self.current_sprite_index = (self.current_sprite_index + 1) % len(self.images)
                self.image = self.images[self.current_sprite_index]

    def draw(self, screen):
        screen.blit(self.image, self.rect)  # Drawing obstacle sprite

        # --- Drawing Red Rectangle around Obstacle ---
        pygame.draw.rect(screen, (255, 0, 0), self.rect, 2)  # Uncomment for debugging