import pygame

class Dino:
    def __init__(self):
        # --- Loading Dino run sprites ---
        self.run_sprites = [
            pygame.image.load("assets/dino/rex_run_0.png"),
            pygame.image.load("assets/dino/rex_run_1.png")
        ]

        # --- Loading Dino jump sprite ---
        self.jump_sprite = pygame.image.load("assets/dino/rex_jump_0.png")

        # --- Loading Dino duck sprites ---
        self.duck_sprites = [
            pygame.image.load("assets/dino/rex_duck_0.png"),
            pygame.image.load("assets/dino/rex_duck_1.png")
        ]

        # In case  of size adjustment
        self.run_sprites[0] = pygame.transform.scale(self.run_sprites[0], (self.run_sprites[0].get_width() // 2, self.run_sprites[0].get_height() // 2))
        self.run_sprites[1] = pygame.transform.scale(self.run_sprites[1], (self.run_sprites[1].get_width() // 2, self.run_sprites[1].get_height() // 2))
        self.jump_sprite = pygame.transform.scale(self.jump_sprite, (self.jump_sprite.get_width() // 2, self.jump_sprite.get_height() // 2))
        self.duck_sprites[0] = pygame.transform.scale(self.duck_sprites[0], (self.duck_sprites[0].get_width() // 2, self.duck_sprites[0].get_height() // 2))
        self.duck_sprites[1] = pygame.transform.scale(self.duck_sprites[1], (self.duck_sprites[1].get_width() // 2, self.duck_sprites[1].get_height() // 2))
        
        self.current_sprite_index = 0 # Starting with the first sprite

        self.image = self.run_sprites[self.current_sprite_index] # Current image in display
        self.dead = False # Initially not dead

        self.rect = self.image.get_rect() # Getting rectangle for positioning
        self.rect.x = 50 # Initial x position (adjusting as needed)
        self.original_rect_bottom = 500 # Storing original bottom for animation reset
        self.rect.bottom = self.original_rect_bottom # Initial y position (adjusting as needed)
        
        # --- Animation timing ---
        self.animation_speed = 5 # Number of frames per sprite
        self.animation_counter = 0 # Counter to control sprite change
        
        # --- Jump related attributes ---
        self.is_jumping = False
        self.jump_velocity = -16 # Initial upward velocity
        
        self.gravity = 0.5 # Gravity strength (controls jump arc)
        self.initial_y_position = self.rect.bottom # Storing intial ground position
        self.y_velocity = 0 # Current vertical velocity

        # --- Ducking related attributes ---
        self.is_ducking = False # Initially not ducking
        self.duck_animation_speed = 5 # Animation speed for ducking
        self.duck_animation_counter = 0
        self.original_rect_bottom = self.rect.bottom # Storing original bottom for ducking reset
        
    def update(self):
        """ Updates Dino animation, jump, and ducking physics."""
        if self.is_jumping:
            self.update_jump()
        elif self.is_ducking:
            self.update_duck_animation()
        else:
            self.update_run_animation()

    def update_jump(self):
        """Updates Dino jump physics."""
        self.rect.height = self.image.get_height() # Updating height of the rect to match the image height
        self.rect.width = self.image.get_width() # Updating width of the rect to match the image width
        self.image = self.jump_sprite # Setting jump sprite
        if self.y_velocity == 0:
            # Setting initial jump velocity when jump starts
            self.y_velocity = self.jump_velocity
        else:
            # Applying gravity to the jump velocity
            self.y_velocity += self.gravity
        
        self.y_velocity += self.gravity # Applying gravity
        self.rect.bottom += self.y_velocity # Updating vertical position
        
        if self.rect.bottom >= self.initial_y_position: # Checking if Dino reached the ground
            self.rect.bottom = self.original_rect_bottom # Resetting to ground position
            self.is_jumping = False # Stopping the jump
            self.y_velocity = 0
            
    def update_duck_animation(self):
        """Updates Dino's ducking animation."""
        self.rect.height = self.duck_sprites[0].get_height() # Adjusting height for ducking
        self.rect.width = self.duck_sprites[0].get_width() # Adjusting width for ducking
        self.rect.bottom = self.original_rect_bottom
        self.duck_animation_counter += 1
        if self.duck_animation_counter >= self.duck_animation_speed:
            self.duck_animation_counter = 0
            self.current_sprite_index = (self.current_sprite_index + 1) % len(self.duck_sprites)
            self.image = self.duck_sprites[self.current_sprite_index]

    def update_run_animation(self):
        """ Updates the dino's animation frame. """
        self.rect.height = self.image.get_height() # Updating height of the rect to match the image height
        self.rect.width = self.image.get_width() # Updating width of the rect to match the image width
        self.rect.bottom = self.original_rect_bottom
        self.animation_counter += 1
        
        if self.animation_counter >= self.animation_speed:
            self.animation_counter = 0 # Resetting the counter
            self.current_sprite_index = (self.current_sprite_index + 1) % len(self.run_sprites) # Cycle to next sprite
            self.image = self.run_sprites[self.current_sprite_index]
    
    def draw(self, screen):
        screen.blit(self.image, self.rect) # Drawing the dino onto the screen

        # --- Drawing Green Rectangle around Dino ---
        pygame.draw.rect(screen, (0, 255, 0), self.rect, 2) # TODO: Uncomment for debugging