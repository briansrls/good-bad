# G-B Survivalists: A Multi-Agent Survival Simulation

This project is a multi-agent simulation demonstrating emergent survival behaviors driven by an internal Good/Bad (G-B) evaluation system. Agents ("Survivalists") navigate a 2D grid world, competing for limited "Food" resources and avoiding "Poison". Their decisions are guided by evaluating the potential impact of actions on their internal G/B scores.

This simulation is based on the EOR (Emotional Optimization Robots) model concepts and is visualized using a web interface.

## Project Structure

- `DESIGN.md`: The detailed design document for the simulation.
- `server.py`: The Flask backend server that runs the simulation logic and serves the web interface.
- `agent.py`: Contains the `Agent` class, including its G-B evaluation logic.
- `environment.py`: Contains the `Environment` class for managing the grid, food, and poison.
- `settings.py`: Stores configurable parameters for the simulation.
- `requirements.txt`: Lists the Python dependencies for this project (Pygame, Flask).
- `main.py`: Original Pygame-based simulation entry point (now superseded by `server.py` for web UI).
- `templates/`: Contains the `index.html` file for the web UI.
- `static/`: Contains `sketch.js` (p5.js client-side logic) and `style.css`.
- `README.md`: This file.

## Setup and Running the Simulation

Follow these steps to set up and run the simulation:

1.  **Clone the repository (if you haven't already) or ensure you have the project files.**

2.  **Navigate to the project directory:**
    ```bash
    cd path/to/your/workspace/demo/multi_agent_survival
    ```

3.  **Create a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies. If you don't have `venv` (it's part of Python 3.3+), ensure your Python installation is up to date.

    ```bash
    # For Linux/macOS
    python3 -m venv venv

    # For Windows
    python -m venv venv
    ```

4.  **Activate the virtual environment:**

    ```bash
    # For Linux/macOS
    source venv/bin/activate

    # For Windows
    .\venv\Scripts\activate
    ```
    Your terminal prompt should now indicate that you are in the virtual environment (e.g., `(venv) Your-User@Your-Machine:...$`).

5.  **Install the required dependencies:**
    With the virtual environment activated, install the packages listed in `requirements.txt` (this includes Pygame for image processing if extended, and Flask for the web server):
    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the simulation server:**
    Once the dependencies are installed, you can run the main simulation server script:
    ```bash
    python server.py
    ```
    This will start the Flask development server. Look for a message like `* Running on http://127.0.0.1:5001/` (or `http://0.0.0.0:5001/`).

7.  **Open the simulation in your web browser:**
    Open your preferred web browser and navigate to the address shown when you started the server, typically:
    [http://localhost:5001](http://localhost:5001) or [http://127.0.0.1:5001](http://127.0.0.1:5001)

8.  **Deactivate the virtual environment (when you're done):**
    When you close the browser tab and stop the server (usually with Ctrl+C in the terminal), you can deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Web UI Controls

-   **Reset:** Resets the simulation to its initial state.
-   **Step:** Advances the simulation by one tick.
-   **Run/Pause:** Starts or pauses the automatic stepping of the simulation.
-   **Speed (FPS) Slider:** Adjusts the number of ticks per second when the simulation is running.
-   **Click on Agent:** Selects an agent on the canvas to display its G/B scores in the info panel.

## Dependencies

-   Python 3.6+ (due to f-strings and Flask/Pygame requirements)
-   Flask (version specified in `requirements.txt`)
-   Pygame (version specified in `requirements.txt` - used by backend core logic, even if not for direct display) 