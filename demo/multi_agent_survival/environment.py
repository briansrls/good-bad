import random
import settings

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Grid: 0 = Empty, 1 = Food, 2 = Poison, or store strings like 'EMPTY', 'FOOD', 'POISON'
        # Using strings for clarity as per Agent.perceive and DESIGN.MD
        self.grid = [['EMPTY' for _ in range(width)] for _ in range(height)]
        self.agents = {} # Store agent objects, perhaps keyed by ID or (x,y) tuple
        self.current_tick = 0

        self._place_initial_items(settings.INITIAL_FOOD_COUNT, 'FOOD')
        self._place_initial_items(settings.INITIAL_POISON_COUNT, 'POISON')

    def _place_initial_items(self, count, item_type):
        for _ in range(count):
            placed = False
            attempts = 0
            while not placed and attempts < self.width * self.height: # Prevent infinite loop
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if self.grid[y][x] == 'EMPTY': # Ensure cell is empty
                    self.grid[y][x] = item_type
                    placed = True
                attempts += 1

    def get_cell_content(self, x, y, ignore_agent_id=None):
        """Returns the content of a cell (e.g., 'EMPTY', 'FOOD', 'POISON', 'AGENT')."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            print(f"ENV.get_cell_content({x},{y}): Returning BOUNDARY", flush=True)
            return 'BOUNDARY' # Should align with Agent's perception
        
        grid_val = self.grid[y][x]
        # print(f"ENV.get_cell_content({x},{y}): Raw grid value is '{grid_val}'", flush=True) # Too verbose now

        # Check for agents at this location first, unless we are ignoring a specific agent
        for agent_id_loop, agent_obj in self.agents.items():
            if agent_obj.is_alive() and agent_obj.x == x and agent_obj.y == y:
                if ignore_agent_id and agent_id_loop == ignore_agent_id:
                    continue # Skip the agent we want to ignore (e.g., the one asking)
                # print(f"ENV.get_cell_content({x},{y}): Other Agent {agent_id_loop} found. Returning AGENT", flush=True)
                return 'AGENT' 
        
        # print(f"ENV.get_cell_content({x},{y}): No other agent found, returning grid value '{grid_val}'", flush=True)
        return grid_val

    def consume_item(self, x, y):
        """Called by an agent when it consumes an item at x, y."""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Item is consumed, cell becomes empty. 
            # The agent's G/B score update is handled in agent.py
            if self.grid[y][x] == 'FOOD' or self.grid[y][x] == 'POISON':
                self.grid[y][x] = 'EMPTY'
                # print(f"Item at ({x},{y}) consumed.") # For debugging
            # else: # Trying to consume from empty or agent cell - should not happen if logic is correct
                # print(f"Warning: Attempt to consume non-item at ({x},{y}): {self.grid[y][x]}")

    def add_agent(self, agent):
        if agent.id not in self.agents:
            self.agents[agent.id] = agent
            # Mark agent's initial position. Note: get_cell_content handles returning 'AGENT'
            # self.grid[agent.y][agent.x] = 'AGENT' # This might be problematic if we want to know underlying item
                                               # Decision: get_cell_content will check agent list separately.
        else:
            print(f"Warning: Agent {agent.id} already in environment.")

    def remove_agent(self, agent_id):
        if agent_id in self.agents:
            # agent_obj = self.agents[agent_id]
            # If we were marking grid with 'AGENT', clear it: self.grid[agent_obj.y][agent_obj.x] = 'EMPTY'
            del self.agents[agent_id]
            # print(f"Agent {agent_id} removed from environment.")

    def update_environment(self):
        """Handles processes like item replenishment per tick."""
        self.current_tick += 1
        # Replenish food
        if random.random() < settings.FOOD_SPAWN_RATE:
            self._spawn_item_if_space('FOOD', settings.INITIAL_FOOD_COUNT)

        # Replenish poison
        if random.random() < settings.POISON_SPAWN_RATE:
            self._spawn_item_if_space('POISON', settings.INITIAL_POISON_COUNT)

    def _count_items(self, item_type):
        count = 0
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == item_type:
                    count += 1
        return count

    def _spawn_item_if_space(self, item_type, target_count_threshold):
        # Only spawn if current count is below some threshold (e.g. initial count)
        # This is a simple interpretation. Design.md Option 2: "respawn randomly... in empty cells"
        # Does not explicitly say to only spawn if below a threshold, but it's reasonable.
        current_count = self._count_items(item_type)
        if current_count < target_count_threshold: # Simple rule: try to maintain up to initial count
            attempts = 0
            placed = False
            while not placed and attempts < 100: # Try a few times to find an empty spot
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if self.grid[y][x] == 'EMPTY':
                    self.grid[y][x] = item_type
                    placed = True
                attempts += 1

    def get_agent_positions(self):
        """Returns a list of (x,y) for all alive agents for drawing or occupancy checks."""
        positions = []
        for agent_id, agent_obj in self.agents.items():
            if agent_obj.is_alive():
                positions.append((agent_obj.x, agent_obj.y))
        return positions

if __name__ == '__main__':
    env = Environment(settings.GRID_WIDTH, settings.GRID_HEIGHT)
    print(f"Environment created with {env.width}x{env.height} grid.")
    food_count = sum(row.count('FOOD') for row in env.grid)
    poison_count = sum(row.count('POISON') for row in env.grid)
    print(f"Initial Food: {food_count}, Initial Poison: {poison_count}")

    # Test consumption
    # env.grid[5][5] = 'FOOD'
    # print(f"Cell (5,5) before consume: {env.get_cell_content(5,5)}")
    # env.consume_item(5,5)
    # print(f"Cell (5,5) after consume: {env.get_cell_content(5,5)}")

    # Test spawning
    # for _ in range(500):
    #     env.update_environment()
    # food_count = sum(row.count('FOOD') for row in env.grid)
    # poison_count = sum(row.count('POISON') for row in env.grid)
    # print(f"After updates - Food: {food_count}, Poison: {poison_count}, Tick: {env.current_tick}")
    print("Environment class defined. Run main.py for simulation.") 