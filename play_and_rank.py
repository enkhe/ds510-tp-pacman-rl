
import gymnasium as gym
import pygame
import numpy as np
import json
import datetime
from stable_baselines3 import PPO
import os

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (100, 100, 255)

ENV_NAME = "MsPacmanNoFrameskip-v4"
MODEL_PATH = "rl-trained-agents/a2c/MsPacmanNoFrameskip-v4_1/MsPacmanNoFrameskip-v4.zip" 
SCOREBOARD_FILE = "scoreboard.json"

# --- Pygame Initialization ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pacman - Human vs. AI")
font = pygame.font.Font(None, 36)
large_font = pygame.font.Font(None, 74)
clock = pygame.time.Clock()

# --- Initial Validation ---
def validate_model_path():
    """Checks if the model path exists and displays an error if it doesn't."""
    if not os.path.exists(MODEL_PATH):
        screen.fill(BLACK)
        error_text = font.render(f"FATAL: Model not found at {MODEL_PATH}", True, RED)
        info_text = font.render("Please train a model first (e.g., run `train.py`).", True, WHITE)
        screen.blit(error_text, (50, SCREEN_HEIGHT // 2 - 50))
        screen.blit(info_text, (50, SCREEN_HEIGHT // 2))
        pygame.display.flip()
        pygame.time.wait(5000)
        return False
    return True

# --- Scoreboard Functions ---

def load_scores():
    """Loads scores from the scoreboard file, creating it if it doesn't exist."""
    if not os.path.exists(SCOREBOARD_FILE):
        with open(SCOREBOARD_FILE, "w") as f:
            json.dump([], f)
        return []
    try:
        with open(SCOREBOARD_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_score(player_name, score):
    """Saves a new score to the scoreboard."""
    scores = load_scores()
    new_score = {
        "name": player_name,
        "score": score,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    scores.append(new_score)
    # Sort scores descending
    scores.sort(key=lambda x: x["score"], reverse=True)
    with open(SCOREBOARD_FILE, "w") as f:
        json.dump(scores, f, indent=4)

def display_scoreboard():
    """Displays the top 10 scores on the pygame screen."""
    scores = load_scores()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        screen.fill(BLACK)
        title_text = font.render("Top 10 Scores", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 50))

        y_offset = 120
        for i, score in enumerate(scores[:10]):
            rank_text = font.render(f"{i+1}.", True, WHITE)
            name_text = font.render(score['name'], True, GREEN)
            score_text = font.render(str(score['score']), True, WHITE)
            date_text = font.render(score['date'], True, BLUE)
            
            screen.blit(rank_text, (50, y_offset))
            screen.blit(name_text, (100, y_offset))
            screen.blit(score_text, (400, y_offset))
            screen.blit(date_text, (500, y_offset))
            y_offset += 40
            
        info_text = font.render("Press ESC to return to menu", True, WHITE)
        screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, SCREEN_HEIGHT - 50))

        pygame.display.flip()

# --- Gameplay Functions ---

def get_player_name():
    """Gets the player's name via keyboard input on the pygame screen."""
    name = ""
    input_active = True
    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None # Signal to quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                else:
                    name += event.unicode
        
        screen.fill(BLACK)
        prompt_text = font.render("Enter Your Name:", True, WHITE)
        name_text = font.render(name, True, GREEN)
        
        screen.blit(prompt_text, (SCREEN_WIDTH // 2 - prompt_text.get_width() // 2, 200))
        screen.blit(name_text, (SCREEN_WIDTH // 2 - name_text.get_width() // 2, 250))
        
        pygame.display.flip()
        clock.tick(60)
    return name if name else "Anonymous"


def human_play():
    """Main loop for human gameplay."""
    # The environment needs wrappers for Atari games
    from rl_zoo3.utils import get_wrapper_class
    from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
    
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    # Wrap the environment for Atari games
    # This is a simplified version of what rl-zoo does
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
    env = gym.wrappers.FrameStack(env, 4)

    obs, _ = env.reset()
    score = 0
    done = False
    
    action = 0 # Default action: NOOP
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return # Exit to menu
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_DOWN:
                    action = 4
                # Optional: other actions if needed
            elif event.type == pygame.KEYUP:
                action = 0 # NOOP when key is released
        
        # The wrapped env returns a LazyFrame, we need to convert it for the step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

        # Render environment to pygame screen
        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(frame_surface, (SCREEN_WIDTH, SCREEN_HEIGHT - 100)), (0, 0))
        
        # Display score
        score_text = font.render(f"Score: {int(score)}", True, WHITE)
        screen.blit(score_text, (10, SCREEN_HEIGHT - 50))
        
        pygame.display.flip()
        clock.tick(60) # Cap frame rate

    # env.close() is not called here to keep the window open for name entry
    player_name = get_player_name()
    if player_name is None: # User quit from name entry
        env.close()
        return
    save_score(player_name, int(score))
    display_scoreboard()
    env.close() # Close env after scoreboard


def bot_play():
    """Main loop for bot gameplay."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        # Display error on screen
        screen.fill(BLACK)
        error_text = font.render(f"Model not found: {MODEL_PATH}", True, RED)
        screen.blit(error_text, (50, SCREEN_HEIGHT // 2 - 50))
        pygame.display.flip()
        pygame.time.wait(3000)
        return

    # Need to import and use the same wrappers as in the zoo
    from rl_zoo3.utils import get_wrapper_class, get_latest_run_id
    from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
    from stable_baselines3 import A2C

    stats_path = os.path.join(os.path.dirname(MODEL_PATH), ENV_NAME)
    hyperparams = {"env_wrapper": ["stable_baselines3.common.atari_wrappers.AtariWrapper"]}

    # This is a simplified version of the logic in enjoy.py
    def make_env():
        env = gym.make(ENV_NAME, render_mode="rgb_array")
        # Used to be gym.wrappers.Monitor, but we render manually
        return env

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    model = A2C.load(MODEL_PATH, env=env)
    
    obs = env.reset()
    score = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return # Exit to menu

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # In a VecEnv, rewards and dones are arrays
        score += reward[0]

        # Render environment to pygame screen
        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(frame_surface, (SCREEN_WIDTH, SCREEN_HEIGHT - 100)), (0, 0))

        # Display score
        score_text = font.render(f"Score: {int(score)}", True, WHITE)
        screen.blit(score_text, (10, SCREEN_HEIGHT - 50))

        pygame.display.flip()
        clock.tick(60) # Cap frame rate

    # env.close() is not called here to keep the window open for the scoreboard
    save_score("A2C_Agent", int(score))
    display_scoreboard()
    env.close() # Close env after scoreboard


# --- Main Menu ---

def main_menu():
    """Displays the main menu and handles user selection."""
    while True:
        screen.fill(BLACK)
        
        title_text = large_font.render("Pacman Challenge", True, WHITE)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))

        # Menu Options
        human_text = font.render("1. Play as Human", True, WHITE)
        bot_text = font.render("2. Watch AI Play", True, WHITE)
        scores_text = font.render("3. View Scoreboard", True, WHITE)
        quit_text = font.render("4. Quit", True, WHITE)

        screen.blit(human_text, (SCREEN_WIDTH // 2 - human_text.get_width() // 2, 250))
        screen.blit(bot_text, (SCREEN_WIDTH // 2 - bot_text.get_width() // 2, 300))
        screen.blit(scores_text, (SCREEN_WIDTH // 2 - scores_text.get_width() // 2, 350))
        screen.blit(quit_text, (SCREEN_WIDTH // 2 - quit_text.get_width() // 2, 400))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    human_play()
                elif event.key == pygame.K_2:
                    bot_play()
                elif event.key == pygame.K_3:
                    display_scoreboard()
                elif event.key == pygame.K_4 or event.key == pygame.K_ESCAPE:
                    return

if __name__ == "__main__":
    if validate_model_path():
        main_menu()
    pygame.quit()
