import random
import settings
import collections # Import collections for deque

class Agent:
    def __init__(self, agent_id, start_x, start_y, agent_type="GB"):
        self.id = agent_id
        self.x = start_x
        self.y = start_y
        self.current_G = settings.INITIAL_G_SCORE
        self.current_B = settings.INITIAL_B_SCORE
        self.vision_range = settings.AGENT_VISION_RANGE
        self.base_action_space = ['MOVE_NORTH', 'MOVE_SOUTH', 'MOVE_EAST', 'MOVE_WEST', 'STAND_STILL']
        self.action_space = list(self.base_action_space) # Start with base move/still actions
        if settings.ENABLE_COMBAT:
            self.action_space.extend(['ATTACK_NORTH', 'ATTACK_SOUTH', 'ATTACK_EAST', 'ATTACK_WEST'])
        self.history = [] # For potential future use (e.g., L3)
        # L3: Initialize visited trail
        self.visited_trail = collections.deque(maxlen=settings.TRAIL_MAX_LENGTH)
        self.visited_trail.append((start_x, start_y)) # Add starting position to trail
        self.agent_type = agent_type # Store agent type ("GB" or "RANDOM")
        print(f"Agent {self.id} initialized as type: {self.agent_type}", flush=True) # Log agent type

    def is_alive(self):
        return (self.current_G > settings.DEATH_CONDITION_G_THRESHOLD and
                self.current_B < settings.DEATH_CONDITION_B_THRESHOLD)

    def perceive(self, environment_grid):
        """
        Perceives the local environment within the agent's vision range.
        Returns a representation of the perceived local grid.
        """
        perceived_data = []
        # The agent is at the center of its vision.
        # Vision extends 'vision_range' cells in each cardinal direction.
        # So, the view is a square of side length (2 * vision_range + 1)
        view_size = (2 * self.vision_range + 1)
        
        for r_offset in range(-self.vision_range, self.vision_range + 1):
            row_data = []
            for c_offset in range(-self.vision_range, self.vision_range + 1):
                # Calculate the actual grid coordinates
                check_x, check_y = self.x + c_offset, self.y + r_offset
                
                # Check boundaries
                if 0 <= check_x < settings.GRID_WIDTH and 0 <= check_y < settings.GRID_HEIGHT:
                    row_data.append(environment_grid[check_y][check_x]) # Assuming grid[y][x]
                else:
                    row_data.append('BOUNDARY') # Represent out-of-bounds
            perceived_data.append(row_data)
        return perceived_data

    def evaluate_action(self, action, perceived_local_grid, environment):
        """
        Evaluates a single potential action based on G-B principles.
        This is the core L2 logic.
        Returns the utility score for the action.
        """
        next_x, next_y = self.x, self.y
        is_move_action = False
        is_attack_action = False
        attack_target_coord = None

        if action.startswith("MOVE_"):
            is_move_action = True
            if action == 'MOVE_NORTH': next_y -= 1
            elif action == 'MOVE_SOUTH': next_y += 1
            elif action == 'MOVE_EAST': next_x += 1
            elif action == 'MOVE_WEST': next_x -= 1
        elif action.startswith("ATTACK_") and settings.ENABLE_COMBAT:
            is_attack_action = True
            # Target coordinate is adjacent for attack
            attack_target_coord = list(self.x_y_for_action_str(action.replace("ATTACK_", "MOVE_"))) # Get target coord
            # Attacker does not move, so next_x, next_y remain self.x, self.y for attacker's own state change
            # The G/B cost of attack is applied to the attacker staying in their current cell.
        elif action == 'STAND_STILL':
            pass # Position remains the same
        else: # Unknown action, should not happen if action_space is respected
            return -float('inf')

        # --- G/B Impact Estimation ---
        delta_G = 0
        delta_B = 0

        # 1. Impact of target cell (for movement) or attack costs
        if is_move_action:
            if not (0 <= next_x < settings.GRID_WIDTH and 0 <= next_y < settings.GRID_HEIGHT):
                return -float('inf') # Invalid move
            target_cell_content = environment.get_cell_content(next_x, next_y, ignore_agent_id=self.id)
            if target_cell_content == 'FOOD':
                delta_G += settings.FOOD_G_REWARD
            elif target_cell_content == 'POISON':
                delta_G -= settings.POISON_G_PENALTY
                delta_B += settings.POISON_B_COST
            elif target_cell_content == 'AGENT': 
                return -float('inf') # Cannot move into an occupied cell
        
        elif is_attack_action:
            # Costs for attacking are applied to the agent
            delta_G -= settings.FIGHT_G_COST
            delta_B += settings.FIGHT_B_COST
            
            # Check if there is a valid target for the attack
            target_x, target_y = attack_target_coord
            if not (0 <= target_x < settings.GRID_WIDTH and 0 <= target_y < settings.GRID_HEIGHT):
                return -float('inf') # Attacking out of bounds is invalid
            
            # Get content of target cell, this time we *don't* ignore self, we look for *other* agents
            target_cell_actual_content = environment.get_cell_content(target_x, target_y)
            if target_cell_actual_content != 'AGENT':
                return -float('inf') # Cannot attack an empty cell, food, or poison
            
            # If there is an agent, the utility also considers the damage dealt
            # This is a simplified utility adjustment. More complex would be predicting opponent's future state.
            # For now, we add a utility bonus proportional to target damage.
            # This delta_G is for the *attacker's utility calculation*, not their actual G change from attacking.
            delta_G += settings.TARGET_G_DAMAGE * settings.W_OPPONENT_DAMAGE_VALUE 
            # (The idea is that damaging an opponent might indirectly lead to more G for self later)

        # 2. Movement Cost (already handled if is_move_action, for attacks this is part of FIGHT_G_COST)
        if is_move_action:
            delta_G -= settings.MOVEMENT_G_COST
            # MOVEMENT_B_COST is 0, so no change to delta_B here
        
        # 3. Anticipated G/B Score (before passive decay/increase)
        anticipated_G_pre_passive = self.current_G + delta_G
        anticipated_B_pre_passive = self.current_B + delta_B

        # 4. Add Passive Time Step Decay/Increase (applied regardless of action)
        # This is slightly different from 5.2.d in DESIGN.MD which applies it *after* choosing action.
        # Applying it here makes the utility calculation reflect the state *after* the next tick fully resolves.
        final_anticipated_G = anticipated_G_pre_passive - settings.TIME_STEP_G_DECAY
        final_anticipated_B = anticipated_B_pre_passive - settings.TIME_STEP_B_DECAY # Use B_DECAY and subtract
        final_anticipated_B = max(0, final_anticipated_B) # Ensure predicted B doesn't go below 0 for utility calc
        
        # 5. Calculate Action Utility (The "GB Thing")
        # Using dynamic weighting as per 5.2.e
        w_g = settings.W_G
        w_b = settings.W_B

        # Dynamic Weighting (simple example based on current state)
        if self.current_G < settings.INITIAL_G_SCORE / 2: # If G is low
            w_g *= 1.5 # Prioritize G-gain more
        if self.current_B > settings.DEATH_CONDITION_B_THRESHOLD / 2: # If B is high
            w_b *= 1.5 # Prioritize B-avoidance more

        utility = (w_g * final_anticipated_G) - (w_b * final_anticipated_B)
        
        # L3: Apply penalty if next_x, next_y is in the visited_trail
        if (next_x, next_y) in self.visited_trail:
            # More sophisticated: penalty could be higher if more recent
            # For now, a flat penalty if in recent trail (excluding current spot if standing still)
            if not (action == 'STAND_STILL' and next_x == self.x and next_y == self.y):
                 utility -= settings.TRAIL_PENALTY
                 # print(f"AGENT {self.id} penalizing move to ({next_x},{next_y}) due to trail. Utility now: {utility}", flush=True)

        # Prevent going into certain death if possible
        if final_anticipated_G <= settings.DEATH_CONDITION_G_THRESHOLD or \
           final_anticipated_B >= settings.DEATH_CONDITION_B_THRESHOLD:
            # If this action leads to death, make it very unattractive unless all actions lead to death.
            # A small negative number to be chosen over -inf if all other options are invalid.
            utility -= 1_000_000 # Large penalty for death-inducing moves

        return utility

    def choose_action(self, environment):
        """
        Chooses an action. If agent_type is "RANDOM", picks randomly.
        Otherwise, uses G-B evaluation.
        """
        if self.agent_type == "RANDOM":
            # Random walker: choose a random valid action from base moves and stand still.
            possible_actions = []
            for action in self.base_action_space: # Iterate over base_action_space (moves and stand still only)
                next_x, next_y = self.x, self.y
                is_move = True
                if action == 'MOVE_NORTH': next_y -=1
                elif action == 'MOVE_SOUTH': next_y +=1
                elif action == 'MOVE_EAST': next_x +=1
                elif action == 'MOVE_WEST': next_x -=1
                elif action == 'STAND_STILL': 
                    is_move = False # Standing still is always possible in terms of bounds
                    possible_actions.append(action)
                    continue
                
                if is_move: # Only check bounds for actual move actions
                    if (0 <= next_x < settings.GRID_WIDTH and 
                        0 <= next_y < settings.GRID_HEIGHT):
                        possible_actions.append(action)
            
            if not possible_actions: 
                 # This should ideally not be reached if STAND_STILL is always an option
                 # but as a safeguard if base_action_space was empty or STAND_STILL was removed.
                 return 'STAND_STILL' 
            
            chosen_action = random.choice(possible_actions)
            # Random walkers don't have "utility" in the G-B sense for this history log
            self.history.append({
                'tick': environment.current_tick if hasattr(environment, 'current_tick') else -1,
                'G': self.current_G, 
                'B': self.current_B,
                'action': chosen_action,
                'utility': 0, # Placeholder for random agents
                'all_utilities': {} # Placeholder
            })
            return chosen_action

        # --- Original G-B Agent Logic ---
        best_action = None
        max_utility = -float('inf')
        action_utilities = {}

        for action in self.action_space:
            # Note: The original 'perceive' method is not directly used by 'evaluate_action' as designed.
            # 'evaluate_action' directly checks the 'environment' for next cell content.
            # This is fine, just a note on the flow.
            utility = self.evaluate_action(action, None, environment) # Pass environment object
            action_utilities[action] = utility
            if utility > max_utility:
                max_utility = utility
                best_action = action
        
        # Handle ties or low utility: pick randomly among bests or explore
        best_actions = [act for act, util in action_utilities.items() if util == max_utility]
        
        if not best_actions or max_utility == -float('inf'): # No valid actions or all are terrible
            # Fallback to random valid move if possible, or stand still
            # For now, let's just stand still if no good options. A better fallback is needed.
            # This check for validity (not moving into boundary/other agent) should be more robust.
            # For now, evaluate_action handles boundary by returning -inf. Agent collision needs specific handling.
             possible_moves = []
             for act_check in self.action_space:
                 if act_check == 'STAND_STILL':
                     possible_moves.append(act_check)
                     continue
                 next_x_check, next_y_check = self.x, self.y
                 if act_check == 'MOVE_NORTH': next_y_check -=1
                 elif act_check == 'MOVE_SOUTH': next_y_check +=1
                 elif act_check == 'MOVE_EAST': next_x_check +=1
                 elif act_check == 'MOVE_WEST': next_x_check -=1
                 
                 if 0 <= next_x_check < settings.GRID_WIDTH and 0 <= next_y_check < settings.GRID_HEIGHT:
                     if environment.get_cell_content(next_x_check, next_y_check) != 'AGENT': # Simple check
                         possible_moves.append(act_check)
             if possible_moves:
                 return random.choice(possible_moves)
             else: # Should not happen if stand still is always an option for evaluate_action
                 return 'STAND_STILL'


        chosen_action = random.choice(best_actions)
        self.history.append({
            'tick': environment.current_tick if hasattr(environment, 'current_tick') else -1, #Requires environment to have tick
            'G': self.current_G, 
            'B': self.current_B,
            'action': chosen_action,
            'utility': max_utility,
            'all_utilities': action_utilities
        })
        return chosen_action

    def update_state(self, action_taken, environment, claimed_food_cells_this_tick):
        """
        Updates the agent's G/B scores and position based on the action taken and environment.
        Also applies passive G/B decay.
        `claimed_food_cells_this_tick` is a set of (x,y) tuples for food already eaten this tick.
        """
        combat_event_info = None # Initialize to None

        # 1. Apply movement and get content of the new cell
        is_move_action = False
        prev_x, prev_y = self.x, self.y 

        if action_taken.startswith("MOVE_"):
            is_move_action = True
            if action_taken == 'MOVE_NORTH': self.y -= 1
            elif action_taken == 'MOVE_SOUTH': self.y += 1
            elif action_taken == 'MOVE_EAST': self.x += 1
            elif action_taken == 'MOVE_WEST': self.x -= 1
        elif action_taken.startswith("ATTACK_") and settings.ENABLE_COMBAT:
            # Attacker does not move. Costs are applied, and target is affected.
            self.current_G -= settings.FIGHT_G_COST
            self.current_B += settings.FIGHT_B_COST
            print(f"AGENT {self.id} ATTACKS {action_taken}. G_cost:{settings.FIGHT_G_COST}, B_cost:{settings.FIGHT_B_COST}. New G:{self.current_G}, B:{self.current_B}", flush=True)

            target_coord = self.x_y_for_action_str(action_taken.replace("ATTACK_", "MOVE_"))
            # We need to ask the environment for the agent object at target_coord
            # This requires a new environment method or passing all agents to update_state.
            # For now, we'll assume server.py will handle finding the target agent and applying damage.
            # Agent.update_state will just signal that an attack happened and where.
            combat_event_info = {
                'attacker_id': self.id,
                'target_coord': target_coord,
                'type': 'attack' # Could be more specific later
            }
        
        # Ensure agent stays within bounds (should be guaranteed by valid action choice)
        self.x = max(0, min(self.x, settings.GRID_WIDTH - 1))
        self.y = max(0, min(self.y, settings.GRID_HEIGHT - 1))

        # 2. G/B impact from movement
        if is_move_action:
            self.current_G -= settings.MOVEMENT_G_COST
            self.current_B += settings.MOVEMENT_B_COST

        # 3. G/B impact from cell content (if moved or stood still on it)
        # Agent consumes item if it moves onto its cell, or if it stands still on it.
        # The environment should handle removing the item.
        # print(f"AGENT {self.id} in update_state: Checking cell ({self.x},{self.y}) for consumption", flush=True) # Too verbose now
        cell_content = environment.get_cell_content(self.x, self.y, ignore_agent_id=self.id) # Pass self.id to ignore self
        # print(f"AGENT {self.id} in update_state: Raw content of ({self.x},{self.y}) (ignoring self) is '{cell_content}'", flush=True) # Too verbose now
        item_consumed_this_tick = False

        if cell_content == 'FOOD':
            current_cell_coords = (self.x, self.y)
            if current_cell_coords not in claimed_food_cells_this_tick:
                print(f"AGENT {self.id} CONSUMING FOOD at {current_cell_coords}. Prev G: {self.current_G}", flush=True)
                self.current_G += settings.FOOD_G_REWARD
                print(f"AGENT {self.id} POST-FOOD G: {self.current_G}", flush=True)
                environment.consume_item(self.x, self.y) # Tell environment to remove food
                claimed_food_cells_this_tick.add(current_cell_coords) # Claim this food cell for this tick
                item_consumed_this_tick = True
            else:
                print(f"AGENT {self.id} at {current_cell_coords}: Food already claimed this tick.", flush=True)
        elif cell_content == 'POISON':
            # For poison, we might not need a claiming system, as multiple agents can suffer from it.
            # Or, if poison is also a one-time consumable item, it would need similar logic.
            # Assuming poison is not "claimed" and affects all who land on it before it's removed (if it is removed).
            # The current environment.consume_item will remove it after the first agent hits it.
            # If multiple agents hit poison that is consumed, this needs similar claiming as food.
            # For now, let's assume poison is consumed by the first agent to process it this tick.
            current_cell_coords = (self.x, self.y) # For consistency if we add claiming to poison
            # To make poison also claimable (if it's a single consumable item):
            # if current_cell_coords not in claimed_poison_cells_this_tick: # (requires new set for poison)
            print(f"AGENT {self.id} CONSUMING POISON at {current_cell_coords}. Prev G: {self.current_G}, Prev B: {self.current_B}", flush=True)
            self.current_G -= settings.POISON_G_PENALTY
            self.current_B += settings.POISON_B_COST
            print(f"AGENT {self.id} POST-POISON G: {self.current_G}, B: {self.current_B}", flush=True)
            environment.consume_item(self.x, self.y) # Tell environment to remove poison
                # claimed_poison_cells_this_tick.add(current_cell_coords) # if poison is claimed
            item_consumed_this_tick = True
            # else:
            #     print(f"AGENT {self.id} at {current_cell_coords}: Poison already claimed this tick.", flush=True)
        
        # 4. Apply passive G decay / B increase (now decay) for the time step
        self.current_G -= settings.TIME_STEP_G_DECAY
        self.current_B -= settings.TIME_STEP_B_DECAY # Apply B-decay
        self.current_B = max(0, self.current_B) # Ensure B doesn't go below 0

        # 5. Cap G/B scores (optional, but good for preventing runaway scores if needed)
        # For now, let death conditions handle extremes. Max G could be a thing.
        # self.current_G = min(self.current_G, MAX_POSSIBLE_G) 
        
        # L3: Add new position to visited trail if moved
        # (Do this after G/B updates from cell, but before next decision cycle)
        # If the agent stood still, its current position is already the most recent in the trail (or will be added if trail was empty).
        # Only add if it actually moved to a new cell to avoid filling trail with stand_still at same spot.
        if self.x != prev_x or self.y != prev_y:
            self.visited_trail.append((self.x, self.y))
        elif not self.visited_trail or self.visited_trail[-1] != (self.x, self.y): 
            # If it stood still but trail is empty or last entry isn't current pos (e.g. after init)
            self.visited_trail.append((self.x, self.y))
            
        # print(f"Agent {self.id} trail: {list(self.visited_trail)}", flush=True)

        # print(f"Agent {self.id} final state this tick: Pos ({self.x},{self.y}), G:{self.current_G:.1f}, B:{self.current_B:.1f}, Consumed: {cell_content if item_consumed_this_tick else 'None'}", flush=True)

        return combat_event_info # Return event info, or None

    def __str__(self):
        return f"Agent {self.id} at ({self.x},{self.y}) G:{self.current_G:.1f} B:{self.current_B:.1f}"

    # Helper function to get coordinates for a given action string (like MOVE_NORTH)
    def x_y_for_action_str(self, action_string):
        tx, ty = self.x, self.y
        if action_string == 'MOVE_NORTH' or action_string == 'ATTACK_NORTH': ty -= 1
        elif action_string == 'MOVE_SOUTH' or action_string == 'ATTACK_SOUTH': ty += 1
        elif action_string == 'MOVE_EAST' or action_string == 'ATTACK_EAST': tx += 1
        elif action_string == 'MOVE_WEST' or action_string == 'ATTACK_WEST': tx -=1
        return tx, ty

if __name__ == '__main__':
    # Basic test
    # Needs a mock environment to run properly
    print("Agent class defined. Run main.py for simulation.")
    # agent = Agent(1, 5, 5)
    # print(agent)
    # print(f"Is alive: {agent.is_alive()}")
    # # Mock environment grid for perception test
    # mock_grid = [['EMPTY' for _ in range(settings.GRID_WIDTH)] for _ in range(settings.GRID_HEIGHT)]
    # mock_grid[5][6] = 'FOOD'
    # mock_grid[4][5] = 'POISON'
    # perceived = agent.perceive(mock_grid)
    # print("Perceived (5x5 around agent at 5,5):")
    # for row in perceived:
    #     print(row) 