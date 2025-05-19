let gSlider, bSlider, masterSlider;
let dropButton;
let canvas;
let ball = null; // Holds the single ball object, null if no ball is active

const BASE_CANVAS_COLOR = [35, 39, 42]; // #23272A (matches body background)
const GOOD_ILLUMINATION_BASE_COLOR_RGB = [114, 137, 218]; // Blurple (RGB for gradient)
const BAD_ILLUMINATION_BASE_COLOR_RGB = [200, 70, 100];   // A magenta/red (RGB for gradient)
const GRADIENT_MIDPOINT_FACTOR = 0.4; // How far towards center gradient reaches (0.0 to 0.5)

const MAX_GRADIENT_ALPHA_BASE = 0.75; // Target alpha for gradient stop when slider is at 100%
const PULSE_SPEED = 0.03;
const PULSE_MAGNITUDE_RATIO = 0.20; // Glow can vary by +/- 20% of its current base intensity

// Physics constants
const GRAVITY = 0.15; // Kept for reference, but not used directly for Y-axis pull anymore
const MAX_ATTRACTION_FORCE = 0.35; // Significantly reduced max horizontal force
const BALL_RADIUS = 12;
const DAMPING_FACTOR = 0.99; // Slight air resistance
const BOUNCE_FACTOR = -0.6; // How much velocity is reversed on bounce
const SCREEN_CENTER_PULL_STRENGTH = 0.001; // Strength of pull towards screen center (X and Y)

// Constants for violent shake effect
const MAX_VISUAL_SHAKE_OFFSET_PIXELS = 6; // Max visual shake offset in pixels when G/B are high and balanced.

const BALL_BASE_COLOR_RGB = [220, 220, 230]; // A light, slightly cool base for the ball
const GOOD_ZONE_INFLUENCE_X_CUTOFF = 0.7; // Good color influences ball up to 70% of screen width from left
const BAD_ZONE_INFLUENCE_X_START = 0.3;  // Bad color influences ball from 30% of screen width from left

function setup() {
  // Canvas setup to fill window
  canvas = createCanvas(windowWidth, windowHeight);
  let canvasContainer = document.getElementById('canvas-container');
  if (canvasContainer) {
      canvas.parent('canvas-container');
  } else {
      console.error("#canvas-container not found, appending canvas to body.");
      canvas.parent(document.body);
  }
  
  window.addEventListener('resize', () => {
    resizeCanvas(windowWidth, windowHeight);
    if (ball) {
        ball.pos.x = constrain(ball.pos.x, BALL_RADIUS, width - BALL_RADIUS);
        // ball.pos.y = constrain(ball.pos.y, BALL_RADIUS, height - BALL_RADIUS); // Ensure ball y is also constrained if necessary
    }
    if (!isLooping()) redraw();
  });

  // --- Controls Setup ---
  // Parent sliders to their respective placeholder divs in #controls-overlay
  let gSliderContainer = document.querySelector('#controls-overlay .slider-group:nth-child(1)');
  let bSliderContainer = document.querySelector('#controls-overlay .slider-group:nth-child(2)');
  let mSliderContainer = document.querySelector('#controls-overlay .slider-group:nth-child(3)'); // New Master Slider container

  if (!gSliderContainer || !bSliderContainer || !mSliderContainer) {
    console.error("Slider container divs not found in #controls-overlay!"); 
    // Basic fallback if needed
    gSliderContainer = gSliderContainer || createDiv().parent(document.body);
    bSliderContainer = bSliderContainer || createDiv().parent(document.body);
    mSliderContainer = mSliderContainer || createDiv().parent(document.body);
  }
  // Remove placeholder text before adding sliders
  if (gSliderContainer.childNodes.length > 1 && gSliderContainer.lastChild.nodeType !== 1) gSliderContainer.lastChild.remove(); 
  if (bSliderContainer.childNodes.length > 1 && bSliderContainer.lastChild.nodeType !== 1) bSliderContainer.lastChild.remove();
  if (mSliderContainer.childNodes.length > 1 && mSliderContainer.lastChild.nodeType !== 1) mSliderContainer.lastChild.remove(); // For Master slider placeholder

  gSlider = createSlider(0, 100, 0, 1);
  gSlider.parent(gSliderContainer);
  gSlider.input(() => { 
    // When G slider is moved, Master slider does not automatically update.
    // It reflects the last *overall* setting or its own independent value.
    if (!isLooping()) redraw(); 
  });

  bSlider = createSlider(0, 100, 0, 1);
  bSlider.parent(bSliderContainer);
  bSlider.input(() => { 
    // Similar to G slider, B does not update Master.
    if (!isLooping()) redraw(); 
  });

  masterSlider = createSlider(0, 100, 0, 1);
  masterSlider.parent(mSliderContainer);
  masterSlider.input(() => {
    const mVal = masterSlider.value();
    gSlider.value(mVal);
    bSlider.value(mVal);
    if (!isLooping()) redraw();
  });
  
  // Button setup
  dropButton = select('#dropBallButton');
  if (dropButton) {
    dropButton.mousePressed(dropNewBall);
  } else {
    console.error("#dropBallButton not found!");
  }
  
  loop(); // Changed from noLoop() to enable continuous animation
  // redraw(); // Not strictly needed if loop() is called, draw() will run automatically
}

function dropNewBall() {
  ball = {
    pos: createVector(width / 2, BALL_RADIUS + 10),
    vel: createVector(random(-1,1), 0),
    acc: createVector(0, 0),
    radius: BALL_RADIUS,
    color: color(BALL_BASE_COLOR_RGB[0], BALL_BASE_COLOR_RGB[1], BALL_BASE_COLOR_RGB[2])
  };
  if (!isLooping()) redraw(); // Ensure a draw if we were in noLoop mode for some reason
}

function applyForcesToBall() {
  if (!ball) return;
  ball.acc.mult(0); // Reset acceleration each frame

  // 1. General attraction towards the screen center (X and Y)
  let screenCenterX = width / 2;
  let screenCenterY = height / 2;
  let pullToCenterX = (screenCenterX - ball.pos.x) * SCREEN_CENTER_PULL_STRENGTH;
  let pullToCenterY = (screenCenterY - ball.pos.y) * SCREEN_CENTER_PULL_STRENGTH;
  ball.acc.x += pullToCenterX;
  ball.acc.y += pullToCenterY;

  // 2. Attraction forces based on sliders (horizontal)
  let gVal = gSlider.value();
  let bVal = bSlider.value();

  if (gVal > 0) {
    let gStrength = map(gVal, 0, 100, 0, MAX_ATTRACTION_FORCE);
    let targetG = createVector(ball.radius, ball.pos.y); 
    let forceG = p5.Vector.sub(targetG, ball.pos);
    forceG.setMag(gStrength);
    ball.acc.add(forceG); // Adds to X and Y, but target Y is ball.pos.y so Y component is 0
  }

  if (bVal > 0) {
    let bStrength = map(bVal, 0, 100, 0, MAX_ATTRACTION_FORCE);
    let targetB = createVector(width - ball.radius, ball.pos.y);
    let forceB = p5.Vector.sub(targetB, ball.pos);
    forceB.setMag(bStrength);
    ball.acc.add(forceB); // Adds to X and Y, but target Y is ball.pos.y so Y component is 0
  }

  // 3. Emergent shake based on G/B balance and overall intensity
  let normalizedDifference = abs(gVal - bVal) / 100.0; // 0 when equal, 1 when max different
  let balanceFactor = 1.0 - normalizedDifference;      // 1 when equal, 0 when max different
  balanceFactor = pow(balanceFactor, 2); // Squaring makes it more sensitive to balance

  let intensityFactor = (gVal + bVal) / 200.0; // 0 when both 0, 1 when both 100

  // Only apply shake if there's some intensity to avoid tiny shakes when sliders are near zero
  ball.visualShakeAmount = 0; // Initialize for this frame
  if (intensityFactor > 0.05) { // Threshold to prevent shake when sliders are very low
      let currentShakeAmount = balanceFactor * intensityFactor * MAX_VISUAL_SHAKE_OFFSET_PIXELS;
      ball.visualShakeAmount = currentShakeAmount; // Store visual shake amount
  }
}

function updateBallPhysics() {
  if (!ball) return;

  ball.vel.add(ball.acc);
  ball.vel.mult(DAMPING_FACTOR);
  ball.pos.add(ball.vel);

  // Boundary conditions (bounce)
  // Sides
  if (ball.pos.x < ball.radius) {
    ball.pos.x = ball.radius;
    ball.vel.x *= BOUNCE_FACTOR;
  } else if (ball.pos.x > width - ball.radius) {
    ball.pos.x = width - ball.radius;
    ball.vel.x *= BOUNCE_FACTOR;
  }

  // Top (less likely to hit with downward gravity, but good to have)
  if (ball.pos.y < ball.radius) {
    ball.pos.y = ball.radius;
    ball.vel.y *= BOUNCE_FACTOR;
  }

  // Bottom - remove ball if it goes off screen (or just stop it)
  if (ball.pos.y > height + ball.radius * 5) { // Give some leeway before removing
    ball = null; // Remove the ball
  }
}

function drawBall() {
  if (!ball) return;

  let displayX = ball.pos.x;
  let displayY = ball.pos.y;

  if (ball.visualShakeAmount && ball.visualShakeAmount > 0) {
    let shakeEffect = p5.Vector.random2D().mult(ball.visualShakeAmount);
    displayX += shakeEffect.x;
    displayY += shakeEffect.y;
  }

  fill(ball.color);
  stroke(20, 20, 30, 200); // Dark, slightly transparent stroke for visibility
  strokeWeight(1.5);
  ellipse(displayX, displayY, ball.radius * 2, ball.radius * 2);
}

function draw() {
  background(BASE_CANVAS_COLOR[0], BASE_CANVAS_COLOR[1], BASE_CANVAS_COLOR[2]);

  let gVal = gSlider.value();
  let bVal = bSlider.value();

  let midX = width * GRADIENT_MIDPOINT_FACTOR;
  let farMidX = width * (1 - GRADIENT_MIDPOINT_FACTOR);

  // Calculate current pulse factor (same for both sides for synchronization, can be different if desired)
  let pulseFactor = sin(frameCount * PULSE_SPEED);

  // --- Good Zone Illumination (Left Side Gradient) ---
  if (gVal > 0) {
    let baseGoodAlpha = map(gVal, 0, 100, 0, MAX_GRADIENT_ALPHA_BASE);
    let currentGoodAlpha = baseGoodAlpha * (1 + pulseFactor * PULSE_MAGNITUDE_RATIO);
    currentGoodAlpha = constrain(currentGoodAlpha, 0, 1); // Ensure alpha stays within 0-1 for rgba
    
    let gradGood = drawingContext.createLinearGradient(0, 0, midX, 0);
    gradGood.addColorStop(0, `rgba(${GOOD_ILLUMINATION_BASE_COLOR_RGB.join(',')}, ${currentGoodAlpha})`);
    gradGood.addColorStop(1, `rgba(${GOOD_ILLUMINATION_BASE_COLOR_RGB.join(',')}, 0)`);
    drawingContext.fillStyle = gradGood;
    noStroke();
    rect(0, 0, midX, height);
  }

  // --- Bad Zone Illumination (Right Side Gradient) ---
  if (bVal > 0) {
    let baseBadAlpha = map(bVal, 0, 100, 0, MAX_GRADIENT_ALPHA_BASE);
    let currentBadAlpha = baseBadAlpha * (1 + pulseFactor * PULSE_MAGNITUDE_RATIO);
    currentBadAlpha = constrain(currentBadAlpha, 0, 1); // Ensure alpha stays within 0-1

    let gradBad = drawingContext.createLinearGradient(width, 0, farMidX, 0);
    gradBad.addColorStop(0, `rgba(${BAD_ILLUMINATION_BASE_COLOR_RGB.join(',')}, ${currentBadAlpha})`);
    gradBad.addColorStop(1, `rgba(${BAD_ILLUMINATION_BASE_COLOR_RGB.join(',')}, 0)`);
    drawingContext.fillStyle = gradBad;
    noStroke();
    rect(farMidX, 0, width - farMidX, height); 
  }
  
  // --- Ball Physics and Drawing ---
  if (ball) {
    applyForcesToBall();
    updateBallPhysics();
    drawBall();
  }
}

// Ensure p5 instance mode is not accidentally triggered if script is moved/wrapped.
// This script assumes global mode p5.js from CDN. 