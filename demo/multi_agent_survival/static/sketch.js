let canvas;
let simData = null;
let cellSize = 10; // Default, will be updated from server
let gridWidth = 50;
let gridHeight = 50;

let selectedAgent = null;
let isRunning = false;
let loopInterval = null;
let currentFPS = 10;

// p5.js setup function - runs once at the beginning
function setup() {
    let canvasContainer = select('#canvas-container');
    // Initialize canvas with a default size, will resize on first data fetch
    canvas = createCanvas(500, 500);
    canvas.parent(canvasContainer);
    noStroke();
    textAlign(CENTER, CENTER);

    // Initialize UI elements
    select('#reset-button').mousePressed(resetSim);
    select('#step-button').mousePressed(stepSim);
    select('#run-button').mousePressed(runSim);
    select('#pause-button').mousePressed(pauseSim);
    select('#fps-slider').input(updateFPS);

    currentFPS = int(select('#fps-slider').value());
    select('#fps-value').html(currentFPS);

    fetchSimulationState(); // Initial fetch
}

// p5.js draw function - runs repeatedly
function draw() {
    background(240); // Light grey background for the canvas area

    if (simData) {
        cellSize = simData.cell_size || 10;
        gridWidth = simData.grid_width || 50;
        gridHeight = simData.grid_height || 50;

        // Adjust canvas size if needed (e.g. on first load or if settings change)
        let expectedWidth = gridWidth * cellSize;
        let expectedHeight = gridHeight * cellSize;
        if (width !== expectedWidth || height !== expectedHeight) {
            resizeCanvas(expectedWidth, expectedHeight);
        }

        // Draw grid items (Food, Poison)
        for (let r = 0; r < gridHeight; r++) {
            for (let c = 0; c < gridWidth; c++) {
                let x = c * cellSize;
                let y = r * cellSize;
                let cellType = simData.grid[r][c];

                if (cellType === 'FOOD') {
                    fill(0, 200, 0); // Green for Food
                } else if (cellType === 'POISON') {
                    fill(200, 0, 0); // Red for Poison
                } else {
                    fill(255); // White for Empty
                }
                rect(x, y, cellSize, cellSize);
                stroke(220); // Light grey border for cells
                noFill();
                rect(x, y, cellSize, cellSize);
                noStroke();
            }
        }

        // Draw agent trails (before drawing agents themselves, so agents are on top)
        simData.agents.forEach(agent => {
            if (agent.is_alive && agent.trail) {
                const trailMaxLength = settings.TRAIL_MAX_LENGTH || 10; // Use client-side settings as fallback
                agent.trail.forEach((pos, index) => {
                    let trailX = pos[0] * cellSize;
                    let trailY = pos[1] * cellSize;
                    
                    // Calculate alpha based on recency (newer = less transparent)
                    // index 0 is oldest, agent.trail.length-1 is newest.
                    let alpha = map(index, 0, agent.trail.length -1 , 30, 150); // Alpha from 30 (faint) to 150 (more visible)
                    if (agent.trail.length === 1) alpha = 150; // Single point in trail, make it visible
                    
                    fill(255, 255, 0, alpha); // Yellowish with dynamic alpha
                    noStroke(); // No border for trail markers
                    rect(trailX, trailY, cellSize, cellSize);
                });
            }
        });

        // Draw combat visuals (after agents, so they are on top of agents if needed, or adjust layer order)
        if (simData.combat_visuals) {
            simData.combat_visuals.forEach(cv => {
                let effectX = cv.coord[0] * cellSize + cellSize / 2; // Center of the cell
                let effectY = cv.coord[1] * cellSize + cellSize / 2;
                let age = simData.tick - cv.tick_initiated;
                let alpha = map(age, 0, settings.COMBAT_VISUAL_DURATION -1, 255, 50); // Fades out
                
                push(); // Save current drawing style
                strokeWeight(max(1, cellSize * 0.1));
                stroke(255, 0, 0, alpha); // Red, fading
                
                // Simple "X" mark for combat
                let offset = cellSize * 0.3;
                line(effectX - offset, effectY - offset, effectX + offset, effectY + offset);
                line(effectX - offset, effectY + offset, effectX + offset, effectY - offset);
                pop(); // Restore drawing style
            });
        }

        // Draw agents
        simData.agents.forEach(agent => {
            if (agent.is_alive) {
                let agentX = agent.x * cellSize;
                let agentY = agent.y * cellSize;
                
                // Agent base color based on type
                if (agent.agent_type === "RANDOM") {
                    fill(150, 150, 150); // Grey for Random Walkers
                } else {
                    fill(50, 100, 200); // Blue for G-B Agents
                }
                rect(agentX, agentY, cellSize, cellSize);

                // Display Agent ID (number part)
                let agentIdNumber = agent.id.split('_')[1]; // Extracts number from "Agent_X"
                fill(255); // White text for good contrast on blue
                textSize(cellSize * 0.6); // Adjust text size based on cell size
                textAlign(CENTER, CENTER);
                text(agentIdNumber, agentX + cellSize / 2, agentY + cellSize / 2);

                // Draw G/B health bars on the agent square
                const barHeight = Math.max(1, cellSize * 0.15); // Height of each bar
                const barMargin = Math.max(1, cellSize * 0.05); // Margin from top and between bars
                const barWidth = cellSize - (2 * barMargin); // Width of the bars

                // G-score bar (Green)
                // Max G for percentage: settings.INITIAL_G_SCORE * 1.5 (e.g. 50 * 1.5 = 75) seems reasonable from previous logic
                // Or, we could use settings.DEATH_CONDITION_G_THRESHOLD (0) up to a defined max (e.g. INITIAL_G_SCORE * 2)
                // Let's use INITIAL_G_SCORE as the 100% mark for simplicity of "good health"
                let gScorePercent = Math.max(0, Math.min(1, agent.g_score / settings.INITIAL_G_SCORE)); 
                fill(0, 220, 0, 200); // Slightly transparent green
                rect(agentX + barMargin, agentY + barMargin, barWidth * gScorePercent, barHeight);

                // B-score bar (Red)
                // Max B for percentage: settings.DEATH_CONDITION_B_THRESHOLD (e.g. 100)
                let bScorePercent = Math.max(0, Math.min(1, agent.b_score / settings.DEATH_CONDITION_B_THRESHOLD));
                fill(220, 0, 0, 200); // Slightly transparent red
                rect(agentX + barMargin, agentY + barMargin + barHeight + barMargin, barWidth * bScorePercent, barHeight);

                // If this agent is selected, highlight it
                if (selectedAgent && selectedAgent.id === agent.id) {
                    stroke(255, 204, 0); // Yellow highlight
                    strokeWeight(2);
                    noFill();
                    rect(agentX, agentY, cellSize, cellSize);
                    noStroke();
                    strokeWeight(1);
                }
            }
        });

        // Update UI Info Panel
        select('#tick-counter').html(simData.tick);
        select('#alive-agents-counter').html(simData.agents.length);
        if (selectedAgent) {
            const agentData = simData.agents.find(a => a.id === selectedAgent.id);
            if (agentData) {
                select('#agent-id').html(agentData.id);
                select('#agent-g-score').html(agentData.g_score);
                select('#agent-b-score').html(agentData.b_score);
            } else { // Agent might have died
                clearSelectedAgentInfo();
            }
        } else {
            clearSelectedAgentInfo();
        }
    }
}

function fetchSimulationState() {
    fetch('/simulation_state')
        .then(response => response.json())
        .then(data => {
            simData = data;
            // console.log("State updated:", simData);
        })
        .catch(error => console.error('Error fetching simulation state:', error));
}

function resetSim() {
    console.log("Resetting simulation...");
    pauseSim(); // Stop current loop if running
    fetch('/reset', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            selectedAgent = null; // Clear selection
            fetchSimulationState(); // Fetch the new initial state
        })
        .catch(error => console.error('Error resetting simulation:', error));
}

function stepSim() {
    if (isRunning) return; // Don't step if auto-running
    console.log("Stepping simulation...");
    fetch('/step', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            // console.log(data.message, "New tick:", data.tick);
            fetchSimulationState(); // Fetch updated state after step
        })
        .catch(error => console.error('Error stepping simulation:', error));
}

function runSim() {
    if (isRunning) return;
    isRunning = true;
    select('#run-button').hide();
    select('#pause-button').show();
    select('#step-button').attribute('disabled', ''); // Disable step button

    function loop() {
        if (!isRunning) return;
        fetch('/step', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                fetchSimulationState();
            })
            .catch(error => {
                console.error('Error during simulation run step:', error);
                pauseSim(); // Stop if there's an error
            });
    }
    // Clear any existing interval before starting a new one
    if (loopInterval) clearInterval(loopInterval);
    loopInterval = setInterval(loop, 1000 / currentFPS);
}

function pauseSim() {
    if (!isRunning) return;
    isRunning = false;
    select('#run-button').show();
    select('#pause-button').hide();
    select('#step-button').removeAttribute('disabled'); // Enable step button
    if (loopInterval) {
        clearInterval(loopInterval);
        loopInterval = null;
    }
}

function updateFPS() {
    currentFPS = int(this.value());
    select('#fps-value').html(currentFPS);
    if (isRunning) { // If running, restart loop with new FPS
        pauseSim();
        runSim();
    }
}

function mousePressed() {
    if (!simData || !simData.agents) return;
    // Check if click is within canvas bounds
    if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) {
        return; 
    }

    let clickedGridX = floor(mouseX / cellSize);
    let clickedGridY = floor(mouseY / cellSize);

    let foundAgent = null;
    for (const agent of simData.agents) {
        if (agent.x === clickedGridX && agent.y === clickedGridY) {
            foundAgent = agent;
            break;
        }
    }

    if (foundAgent) {
        selectedAgent = foundAgent;
        console.log("Selected Agent:", selectedAgent.id);
    } else {
        selectedAgent = null;
        clearSelectedAgentInfo();
    }
}

function clearSelectedAgentInfo() {
    select('#agent-id').html('N/A');
    select('#agent-g-score').html('N/A');
    select('#agent-b-score').html('N/A');
}

// Assuming settings are not directly available client-side for INITIAL_G_SCORE etc.
// We can either pass them from server or use reasonable defaults for display percentages.
// For now, using implicit knowledge of common values for G/B score ranges.
const settings = { // Minimal mock settings for display logic if not passed from server
    INITIAL_G_SCORE: 50,
    DEATH_CONDITION_B_THRESHOLD: 100,
    TRAIL_MAX_LENGTH: 10, // Added for trail visualization consistency
    COMBAT_VISUAL_DURATION: 3 // Added for combat visualization consistency
};

// Periodically refresh state if not auto-running (e.g., for external changes)
// setInterval(() => {
//     if (!isRunning) {
//         fetchSimulationState();
//     }
// }, 5000); // Refresh every 5 seconds if paused 