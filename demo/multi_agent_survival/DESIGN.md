Okay, let's draft a design document for the "G-B Survivalists" project, with a clear focus on the Good/Bad (G-B) evaluation mechanism inspired by your EOR model.

## Design Document: G-B Survivalists - A Multi-Agent Survival Simulation

**1. Introduction & Overview**

* **Project Goal:** To create a multi-agent simulation demonstrating emergent survival behaviors driven by an internal Good/Bad (G-B) evaluation system inspired by the EOR (Emotional Optimization Robots) model.
* **Core Concept:** Agents ("Survivalists") navigate a 2D grid world, competing for limited "Food" resources (improving their "Good" state) and avoiding "Poison" (worsening their "Bad" state). Their decisions are primarily guided by evaluating the potential impact of actions on their internal G/B scores, reflecting a basic form of "moralized" decision-making for survival.
* **Benchmark Potential:** The system will be designed with the possibility of introducing other AI agent models later for comparative benchmarking.

**2. Core EOR/G-B Principles to be Demonstrated**

* **L0 (Physical Reality/Inputs):** The state of the grid environment (Food, Poison, other agents, terrain) and the agent's own physical position.
* **L1 (Primitive Good/Bad Feeling):** Represented directly by the agent's internal G-score (e.g., energy/health) and B-score (e.g., damage/hunger/stress). Fluctuations in these scores are the agent's fundamental "good" or "bad" feelings.
* **L2 (Rudimentary Self-Awareness & G-B Evaluation for Action):**
    * Agents' actions are not purely reactive but are chosen based on an *evaluation* of their current internal (G,B) state and the *anticipated* G/B consequences of potential actions in their perceived environment.
    * This demonstrates a basic self-referential loop: "My internal state is X (G,B); doing Y in environment Z will likely change my state to X'. Is X' better or worse for my survival?"
* **"Forced Moralization":** All significant perceptions and potential actions are filtered through the G/B evaluation lens – "Is this good for my survival? Is this bad for my survival?"

**3. Game Environment Design**

* **3.1. Grid Specifications:**
    * Type: 2D discrete grid.
    * Size: Configurable (e.g., default 50x50 cells).
    * Cell States: Empty, Food, Poison, Agent.
    * Boundaries: Walled (agents cannot move outside).
* **3.2. Resource & Hazard Types:**
    * **Food:**
        * Visual Representation: E.g., green square.
        * G-Impact: Increases agent's G-score by `FOOD_G_REWARD` (e.g., +20 G).
        * B-Impact: None or negligible (e.g., minimal B-cost for "digestion" if desired).
    * **Poison (Optional but Recommended):**
        * Visual Representation: E.g., red square.
        * G-Impact: None or decreases G-score by `POISON_G_PENALTY` (e.g., -5 G).
        * B-Impact: Increases agent's B-score by `POISON_B_COST` (e.g., +30 B).
* **3.3. Spawning & Depletion Mechanics:**
    * **Initial Placement:** Configurable number of Food and Poison items placed randomly at simulation start.
    * **Replenishment (Food):**
        * Option 1 (Scarcity Focus): No replenishment, or very slow random spawning.
        * Option 2 (Sustained Play): Food items respawn randomly at a configurable rate (`FOOD_SPAWN_RATE`) in empty cells.
    * **Replenishment (Poison):** Similar to Food, likely at a lower rate (`POISON_SPAWN_RATE`).
    * **Depletion:** When an agent moves onto a Food or Poison cell, the item is consumed and removed from the grid.
* **3.4. World Parameters (Configurable):**
    * `MAX_AGENTS`: Maximum number of agents in the simulation.
    * `INITIAL_G_SCORE`, `INITIAL_B_SCORE`: Starting scores for agents.
    * `DEATH_CONDITION_G_THRESHOLD`: G-score below which agent dies (e.g., <= 0).
    * `DEATH_CONDITION_B_THRESHOLD`: B-score above which agent dies (e.g., >= 100).
    * `MOVEMENT_G_COST`: G-score reduction per move (simulating energy use).
    * `MOVEMENT_B_COST`: B-score increase per move (simulating fatigue).
    * `TIME_STEP_G_DECAY / B_INCREASE`: Passive G decay or B increase per simulation tick to force action.

**4. Agent (G-B Survivalist) Design**

* **4.1. Identity:** Each agent has a unique ID.
* **4.2. Internal State (L1):**
    * Current G-score: `current_G`
    * Current B-score: `current_B`
    * Current Position: `(x, y)`
* **4.3. Sensory Perception (Input for L2 Evaluation):**
    * Vision Range: Configurable square radius (e.g., 2 cells in each direction, making a 5x5 view).
    * Perceived Data: For each cell in vision range: cell type (Empty, Food, Poison, OtherAgentID).
* **4.4. Action Space:**
    * `MOVE_NORTH`, `MOVE_SOUTH`, `MOVE_EAST`, `MOVE_WEST`, `STAND_STILL`.
    * One action per simulation tick.
* **4.5. "Death" Condition:**
    * If `current_G` <= `DEATH_CONDITION_G_THRESHOLD` OR `current_B` >= `DEATH_CONDITION_B_THRESHOLD`.
    * Upon death, the agent is removed from the grid.
    * (Optional) Dead agents might leave behind a small amount of Food or Poison.

**5. G-B Evaluation Logic & Decision-Making (L2 Core)**

* **5.1. Goal:** To choose an action that maximizes anticipated G-score and minimizes anticipated B-score, weighted by the agent's current internal G/B state.
* **5.2. Evaluation Process per Tick:**
    1.  **Perceive Environment:** Get local grid information.
    2.  **Consider Potential Actions:** For each action in the Action Space (N, S, E, W, Still):
        a.  **Predict Future Cell:** Determine the cell `(next_x, next_y)` the action would lead to.
        b.  **Estimate Direct G/B Impact of Target Cell:**
            * If `(next_x, next_y)` contains Food: `delta_G = FOOD_G_REWARD`, `delta_B = 0`.
            * If `(next_x, next_y)` contains Poison: `delta_G = -POISON_G_PENALTY`, `delta_B = POISON_B_COST`.
            * If `(next_x, next_y)` is Empty: `delta_G = 0`, `delta_B = 0`.
            * If `(next_x, next_y)` is an Obstacle/Boundary/Other Agent (basic): Action is invalid or has high B cost.
        c.  **Add Movement Cost:**
            * If action is a move: `delta_G -= MOVEMENT_G_COST`, `delta_B += MOVEMENT_B_COST`.
            * If action is Stand Still: No additional movement cost (but time step decay still applies).
        d.  **Calculate Anticipated G/B Score for this Action:**
            * `anticipated_G = current_G + delta_G - TIME_STEP_G_DECAY`
            * `anticipated_B = current_B + delta_B + TIME_STEP_B_INCREASE`
        e.  **Calculate Action Utility (The "GB Thing"):**
            * This is the core G-B evaluation. Example utility function:
                `Utility = (w_g * anticipated_G) - (w_b * anticipated_B)`
                * `w_g` and `w_b` are weights.
            * **Dynamic Weighting based on Internal State (L2 Self-Referential Influence):**
                * If `current_G` is low / `current_B` is high (desperation): Increase `w_g` for actions leading to Food, increase `w_b` for actions leading away from Poison (making G more attractive and B more repulsive).
                * If `current_G` is high / `current_B` is low (satiation/safety): `w_g` and `w_b` might be more balanced, or exploration (randomness) might be favored if no high-utility actions are present.
                * This models the "L2 system making L1 feel better."
    3.  **Action Selection:**
        * Choose the action with the highest `Utility`.
        * In case of ties, or if all utilities are below a threshold, pick randomly among tied bests or explore (e.g., random valid move). This prevents agents from getting stuck.
* **5.3. Handling Competition (Basic):**
    * The environment resolves conflicts: if two agents attempt to move to the same Food cell in the same tick, one (e.g., randomly chosen, or based on an "initiative" stat) gets it, the other finds the cell empty or occupied.
    * Agents currently only perceive other agents as obstacles or non-interactive entities. Future enhancements could allow them to "evaluate" other agents.

**6. Benchmarking Considerations (Phase 2)**

* An API will be defined (`get_observation`, `choose_action`) to allow different agent controller models (Rule-based, Q-Learning, etc.) to be plugged into the same environment for direct comparison against the G-B Survivalist model.
* Metrics for comparison: average lifespan, total Food collected, final population count, G/B score trajectories.

**7. Visualization & User Interface (UI)**

* **7.1. Real-time Grid Display:**
    * Agents: Distinct sprites/colors. Color intensity or an aura could represent `current_G` (brighter = higher G) or `current_B` (more intense red = higher B).
    * Food/Poison: Clear visual distinction.
* **7.2. Information Panel:**
    * Selected Agent View: Display clicked agent's ID, `current_G`, `current_B`, recent action/utility calculation.
    * Global View: Number of agents, average G/B, simulation time/tick, Food/Poison counts.
* **7.3. Controls:**
    * Start, Pause, Reset, Step-forward-one-tick.
    * Simulation speed slider.
    * Input fields for key world parameters (allow modification before starting a new simulation).

**8. Technical Stack (Suggestions)**

* **Python:**
    * **Pygame:** For 2D graphics, event handling.
    * **Mesa:** For agent-based model structure (grid, agent scheduling, data collection).
    * (Or custom classes for agents and grid if preferred for simplicity).
* **JavaScript (Web-Based):**
    * **p5.js or Phaser:** For 2D canvas rendering and interactivity.
    * HTML/CSS for UI elements.

**9. Success Metrics & Observable Behaviors (Demonstrating EOR/G-B Principles)**

* **Survival:** G-B agents consistently survive longer than random walkers or naive greedy agents (that ignore Poison or their own state).
* **Adaptive Behavior:** Agents change their risk-taking for Food based on their internal G/B state (e.g., more desperate when G is low).
* **Clear G/B Driven Action:** Visualization shows agents moving towards cells that improve their G/B utility and away from those that worsen it.
* **Emergent Patterns:**
    * Do agents "cluster" around Food sources?
    * Do they "learn" to avoid dangerous zones with high Poison density (even without explicit memory, just by G/B evaluation of local areas)?
    * How do they behave under extreme scarcity or danger?

**10. Potential Future Enhancements**

* **Rudimentary L3 (Pattern Recognition):** Agents remember locations of consistently good/bad G/B outcomes.
* **Social G-B Evaluation:** Agents assign G/B scores to other agents based on their observed behavior (e.g., "Agent X often takes Food I was moving towards" -> lower G/higher B for Agent X in future utility calculations for cooperative actions).
* **Communication:** Agents emit simple G/B signals.
* More sophisticated learning algorithms for the decision-making utility function.

This design document provides a solid foundation for your "G-B Survivalists" demo, ensuring the G-B evaluation mechanism is central and reflects the principles of your EOR model.