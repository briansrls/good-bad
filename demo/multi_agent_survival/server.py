print("SERVER.PY: Script started", flush=True) # Diagnostic print

from flask import Flask, jsonify, render_template
import settings
from environment import Environment
from agent import Agent
import random
import threading
import time

print("SERVER.PY: Imports successful", flush=True) # Diagnostic print

app = Flask(__name__, static_folder='static', template_folder='templates')
print("SERVER.PY: Flask app initialized", flush=True) # Diagnostic print

# Global simulation objects
# We need to be careful with multi-threading if Flask handles requests concurrently by default
# For simplicity, ensure simulation updates are atomic or synchronized if needed.
print("SERVER.PY: About to initialize global sim_env", flush=True) # Diagnostic print
sim_env = Environment(settings.GRID_WIDTH, settings.GRID_HEIGHT)
print("SERVER.PY: Global sim_env initialized", flush=True) # Diagnostic print
sim_agents_list = []
sim_lock = threading.RLock() # To protect access to shared simulation data. Changed to RLock.
active_combat_visuals = [] # NEW: List to store active combat visuals {coord: (x,y), tick_initiated: t, attacker_id: id}
print("SERVER.PY: Global sim variables (list, lock, combat_visuals) initialized", flush=True) # Diagnostic print

next_agent_id_counter = 0

def get_next_agent_id():
    global next_agent_id_counter
    # print("SERVER.PY: get_next_agent_id called", flush=True) # Can be too verbose
    with sim_lock:
        id_val = next_agent_id_counter
        next_agent_id_counter += 1
        # print(f"SERVER.PY: Assigning Agent ID: Agent_{id_val}", flush=True) # Can be too verbose
        return f"Agent_{id_val}"

def initialize_simulation():
    global sim_env, sim_agents_list, next_agent_id_counter
    print("SERVER.PY: initialize_simulation() called", flush=True) # Diagnostic print
    with sim_lock:
        print("SERVER.PY: initialize_simulation() acquired sim_lock", flush=True) # Diagnostic print
        next_agent_id_counter = 0
        sim_env = Environment(settings.GRID_WIDTH, settings.GRID_HEIGHT)
        print(f"SERVER.PY: initialize_simulation() created new Environment ({settings.GRID_WIDTH}x{settings.GRID_HEIGHT})", flush=True) # Diagnostic print
        sim_agents_list = []
        print(f"SERVER.PY: initialize_simulation() starting agent creation loop for {settings.MAX_AGENTS} agents", flush=True) # Diagnostic print
        for i in range(settings.MAX_AGENTS):
            # print(f"SERVER.PY: initialize_simulation() trying to place agent {i+1}/{settings.MAX_AGENTS}", flush=True) # Can be too verbose
            placed = False
            attempts = 0
            max_attempts = settings.GRID_WIDTH * settings.GRID_HEIGHT * 2 # Increased max attempts
            while not placed and attempts < max_attempts: # Increased max attempts
                start_x = random.randint(0, settings.GRID_WIDTH - 1)
                start_y = random.randint(0, settings.GRID_HEIGHT - 1)
                if sim_env.get_cell_content(start_x, start_y) == 'EMPTY':
                    agent_id_val = get_next_agent_id()
                    # Determine agent type
                    agent_type = "GB"
                    if random.random() < settings.RANDOM_WALKER_PROBABILITY:
                        agent_type = "RANDOM"
                    
                    new_agent = Agent(agent_id=agent_id_val, start_x=start_x, start_y=start_y, agent_type=agent_type)
                    sim_agents_list.append(new_agent)
                    sim_env.add_agent(new_agent)
                    placed = True
                    # print(f"SERVER.PY: initialize_simulation() Placed agent {agent_id} at ({start_x},{start_y})", flush=True) # Can be too verbose
                attempts += 1
            if not placed:
                 print(f"SERVER.PY: WARNING - initialize_simulation() FAILED to place agent {i+1} after {max_attempts} attempts. Grid might be too full or unplaceable.", flush=True)

        print("SERVER.PY: initialize_simulation() finished agent creation loop", flush=True) # Diagnostic print
        print("SERVER.PY: Simulation Initialized (from initialize_simulation func)", flush=True) # Diagnostic print
    print("SERVER.PY: initialize_simulation() released sim_lock", flush=True) # Diagnostic print

def simulation_step():
    with sim_lock:
        # print("SERVER.PY: simulation_step() acquired sim_lock", flush=True) # Can be too verbose
        agent_actions = {}
        for agent in sim_agents_list:
            if agent.is_alive():
                action = agent.choose_action(sim_env)
                agent_actions[agent.id] = action

        agents_to_remove_ids = []
        # current_live_agents = [] # This list was created but not used meaningfully, removing for now
        
        claimed_food_cells_this_tick = set() # NEW: Track food cells claimed this tick
        new_combat_events_this_tick = [] # Store combat events from this tick

        for agent in sim_agents_list:
            if agent.is_alive():
                if agent.id in agent_actions:
                    combat_event = agent.update_state(agent_actions[agent.id], sim_env, claimed_food_cells_this_tick)
                    if combat_event:
                        new_combat_events_this_tick.append(combat_event)
                if not agent.is_alive():
                    agents_to_remove_ids.append(agent.id)
                # else:
                #    current_live_agents.append(agent)
            elif agent.id not in agents_to_remove_ids: # Already dead, ensure it's marked for removal
                 agents_to_remove_ids.append(agent.id)

        # Update agent list - filter out those marked for removal by ID
        sim_agents_list[:] = [agent for agent in sim_agents_list if agent.id not in agents_to_remove_ids]
        
        # Remove from environment object as well
        for agent_id in agents_to_remove_ids:
            sim_env.remove_agent(agent_id)
            # print(f"{agent_id} died and was removed from sim_env.")

        # AFTER agent updates, process combat events (apply damage to targets)
        for event in new_combat_events_this_tick:
            if event['type'] == 'attack':
                target_x, target_y = event['target_coord']
                # Find the agent at the target coordinate
                target_agent = None
                for agent_obj in sim_agents_list: # Check current list of (potentially just updated) agents
                    if agent_obj.x == target_x and agent_obj.y == target_y and agent_obj.id != event['attacker_id']:
                        target_agent = agent_obj
                        break
                
                if target_agent and target_agent.is_alive():
                    prev_g = target_agent.current_G
                    target_agent.current_G -= settings.TARGET_G_DAMAGE
                    print(f"COMBAT: Agent {event['attacker_id']} attacked Agent {target_agent.id} at ({target_x},{target_y}). Target G: {prev_g} -> {target_agent.current_G}", flush=True)
                    # Add to combat visuals
                    active_combat_visuals.append({
                        'coord': (target_x, target_y),
                        'tick_initiated': sim_env.current_tick,
                        'attacker_id': event['attacker_id']
                    })
                elif target_agent: # Target was found but might have died from this or other causes before damage application
                    print(f"COMBAT: Agent {event['attacker_id']} attacked, but target {target_agent.id} at ({target_x},{target_y}) is no longer alive or targetable for damage.", flush=True)
                else:
                    print(f"COMBAT: Agent {event['attacker_id']} attacked ({target_x},{target_y}), but no target agent found there.", flush=True)

        # Remove expired combat visuals
        current_tick = sim_env.current_tick
        active_combat_visuals[:] = [viz for viz in active_combat_visuals 
                                    if current_tick < viz['tick_initiated'] + settings.COMBAT_VISUAL_DURATION]

        sim_env.update_environment()
        # print(f"Tick: {sim_env.current_tick}, Agents: {len(sim_agents_list)}", flush=True) # Can be too verbose
    # print("SERVER.PY: simulation_step() released sim_lock", flush=True) # Can be too verbose

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulation_state', methods=['GET'])
def get_simulation_state():
    with sim_lock:
        grid_state = [[sim_env.grid[r][c] for c in range(sim_env.width)] for r in range(sim_env.height)]
        agents_data = [
            {
                'id': agent.id, 
                'x': agent.x, 
                'y': agent.y, 
                'g_score': round(agent.current_G, 1),
                'b_score': round(agent.current_B, 1),
                'is_alive': agent.is_alive(),
                'trail': list(agent.visited_trail),
                'agent_type': agent.agent_type # Add agent_type to state for client
            } for agent in sim_agents_list if agent.is_alive() # Only send alive agents
        ]
        state = {
            'grid': grid_state,
            'agents': agents_data,
            'tick': sim_env.current_tick,
            'grid_width': sim_env.width,
            'grid_height': sim_env.height,
            'cell_size': settings.CELL_SIZE, # Client might need this for drawing
            'combat_visuals': active_combat_visuals # Send active combat visuals
        }
        return jsonify(state)

@app.route('/step', methods=['POST'])
def step_simulation_endpoint():
    simulation_step()
    return jsonify({'message': 'Simulation stepped', 'tick': sim_env.current_tick})

@app.route('/reset', methods=['POST'])
def reset_simulation_endpoint():
    initialize_simulation()
    return jsonify({'message': 'Simulation reset'})


# Background thread for simulation loop
stop_event = threading.Event()
simulation_thread = None

def run_simulation_loop():
    while not stop_event.is_set():
        simulation_step()
        time.sleep(1.0 / settings.FPS) # Control simulation speed

if __name__ == '__main__':
    print("SERVER.PY: __main__ block started", flush=True) # Diagnostic print
    initialize_simulation() # Initialize once on start
    print("SERVER.PY: initialize_simulation() call completed in __main__", flush=True) # Diagnostic print
    
    # Start the background simulation thread
    # Comment out if you want manual stepping via API only initially
    # simulation_thread = threading.Thread(target=run_simulation_loop, daemon=True)
    # simulation_thread.start()
    # print("SERVER.PY: Background simulation thread started (if uncommented)", flush=True) # Diagnostic print
    
    print("SERVER.PY: About to call app.run()", flush=True) # Diagnostic print
    app.run(debug=True, host='0.0.0.0', port=5001) # Using port 5001 to avoid common conflicts
    print("SERVER.PY: app.run() has exited (should not happen during normal operation unless server is stopped)", flush=True) # Diagnostic print
    
    # When Flask server stops (e.g. Ctrl+C)
    # stop_event.set()
    # if simulation_thread:
    #     simulation_thread.join()
    # print("SERVER.PY: Simulation stopped (if background thread was running)", flush=True) # Diagnostic print 