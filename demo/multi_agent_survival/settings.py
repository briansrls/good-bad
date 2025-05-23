# Screen dimensions for Pygame
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CELL_SIZE = 10 # Size of each grid cell in pixels
GRID_WIDTH = 25 # Was 50. Cells
GRID_HEIGHT = 25 # Was 50. Cells

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) # Food
RED = (255, 0, 0)   # Poison
BLUE = (0, 0, 255)  # Agent

# Simulation Parameters (from DESIGN.MD Section 3.4 & others)
MAX_AGENTS = 10
INITIAL_G_SCORE = 100
INITIAL_B_SCORE = 0
DEATH_CONDITION_G_THRESHOLD = 0
DEATH_CONDITION_B_THRESHOLD = 100

MOVEMENT_G_COST = 0.5
MOVEMENT_B_COST = 0 # Was 1. Movement no longer incurs a B-cost.
TIME_STEP_G_DECAY = 0.4 # Was 0.2. Increasing G decay to encourage movement.
TIME_STEP_B_DECAY = 0.05 # New setting: B-score passively decays over time.

FOOD_G_REWARD = 20
POISON_G_PENALTY = 5 # G-score reduction from poison
POISON_B_COST = 40 # Was 30

INITIAL_FOOD_COUNT = 20 # Was 30
INITIAL_POISON_COUNT = 20 # Was 15

FOOD_SPAWN_RATE = 0.07 # Was 0.1 # Chance per tick to spawn new food if below initial count
POISON_SPAWN_RATE = 0.05 # Chance per tick to spawn new poison

# Agent Parameters (from DESIGN.MD Section 4)
AGENT_VISION_RANGE = 2 # e.g., 2 cells in each direction (5x5 view)

# G-B Evaluation Weights (from DESIGN.MD Section 5.2.e)
W_G = 1.0 # Weight for anticipated G-score
W_B = 1.0 # Weight for anticipated B-score

# L3 Trail Settings (for discouraging re-treading)
TRAIL_MAX_LENGTH = 10  # How many recent steps to remember
TRAIL_PENALTY = 10     # Was 25. Utility penalty for moving to a cell in the recent trail

# Agent Combat Settings
ENABLE_COMBAT = True
FIGHT_G_COST = 2       # G-cost to initiator for performing an attack
FIGHT_B_COST = 30      # B-cost to initiator for performing an attack
TARGET_G_DAMAGE = 25   # G-score damage dealt to the target of an attack
W_OPPONENT_DAMAGE_VALUE = 0.5 # Factor for how much agent values damaging opponent in utility calc
COMBAT_VISUAL_DURATION = 3 # Ticks the combat animation will last
# ATTACK_RANGE = 1     # For now, only adjacent. Can be expanded.

# Agent Type Mix (for benchmarking)
RANDOM_WALKER_PROBABILITY = 0.3 # Probability that a new agent will be a Random Walker

# UI settings
FPS = 10 # Simulation speed (frames per second) 