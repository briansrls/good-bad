# EOR L1 Demo Project Scaffold

A minimal but complete repository to explore paraconsistent (Good,Bad) learning. This scaffold reproduces a toy experiment with synthetic data, organising code into re-usable modules for easy iteration on datasets, models, or loss functions.

## 0. Purpose

This project demonstrates a basic paraconsistent learning setup using a (Good, Bad) output representation for a neural network. It trains two models:
1.  A **Dual Scalar Model**: An MLP that predicts G and B values independently, trained with standard Binary Cross-Entropy loss for each component.
2.  A **(G,B) Model**: An MLP of the same architecture, but trained with a specialized `gb_loss` that includes terms inspired by paraconsistent logic, aiming to better model the relationship and potential overlap between "Good" and "Bad" aspects of a proposition.

The goal is to provide a simple, runnable starting point for experimenting with such concepts.

## 1. Directory Layout

```
g_b_demo/
├── README.md              # This quick‑start guide
├── .gitignore             # Specifies intentionally untracked files
├── environment.yml        # Conda environment specification (CPU)
├── requirements.txt       # pip requirements (used with venv)
├── setup.py               # For making local modules installable
├── data/
│   ├── __init__.py
│   └── generator.py       # Synthetic Knowledge Base + label noise generation
├── models/
│   ├── __init__.py
│   └── gb_model.py        # MLP architecture + (G,B) head and paraconsistent loss
├── train.py               # Script to train both Dual Scalar and G/B models
├── eval.py                # Script to evaluate trained models and generate plots
├── experiments.cfg        # Hyperparameters in one place
├── notebooks/
│   └── 01_quickstart.ipynb # (As per original plan, for interactive exploration)
└── .venv/                 # (Optional) Python virtual environment directory
```

## 2. Setup

You can set up the project environment using either Conda (recommended for managing PyTorch versions consistently) or a Python virtual environment (`venv`).

### 2.1 Using Conda (Recommended)

1.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate eor_l1_demo
    ```
2.  **Install local packages (optional, for notebook usage):**
    If you plan to use the Jupyter notebook and import local modules in an editable way:
    ```bash
    pip install -e .
    ```

### 2.2 Using Python Virtual Environment (`venv`)

1.  **Ensure you have `python3` (version 3.11+ recommended) and `python3-venv` installed.**
    On Debian/Ubuntu, if `python3-venv` is missing:
    ```bash
    sudo apt update
    sudo apt install python3-venv
    ```

2.  **Create a virtual environment (e.g., named `.venv`):**
    From the project root directory (`eor_l1_demo/`):
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    Your terminal prompt should change to indicate the active environment (e.g., `(.venv) your-prompt$`).

4.  **Install dependencies:**
    Once the virtual environment is activated:
    ```bash
    pip install -r requirements.txt
    ```
    Or, if not activated, you can run pip from the venv directly:
    ```bash
    # .venv/bin/pip install -r requirements.txt # (if not activated)
    ```

5.  **Install local packages (optional, for notebook usage):**
    If you plan to use the Jupyter notebook and import local modules in an editable way (run from an activated venv):
    ```bash
    pip install -e .
    ```

## 3. How to Run

Ensure your chosen environment (Conda or venv) is activated for the following commands, or prefix Python commands with `.venv/bin/python` if using an unactivated venv.

### 3.1 Training the Models

The `train.py` script trains both the Dual Scalar model and the G/B model.

*   **To run training (e.g., for 5 epochs):**
    ```bash
    python train.py --epochs 5
    ```
    Models will be saved by default to `runs/demo/`. You can change the save path with the `--save` argument (e.g., `python train.py --epochs 5 --save runs/my_experiment`).

*   **What it does:**
    *   Loads hyperparameters from `experiments.cfg`.
    *   Generates synthetic data using `data/generator.py`:
        *   Creates a small knowledge base with `N_ATOMS`.
        *   Introduces contradictions (`CONTR_RATE`) and label noise (`LABEL_NOISE`).
        *   Encodes atoms into (Good, Bad) target vectors. Unknowns are `(0,0)`, knowns are `(1,0)` or `(0,1)` (before noise).
    *   Trains the Dual Scalar model using a combined Binary Cross-Entropy loss on G and B components.
    *   Trains the (G,B) model using the custom `gb_loss` function.
    *   Prints loss values for both models at each epoch.
    *   Saves the trained model weights (`scalar_model.pth`, `gb_model.pth`) to the specified save directory.

*   **Reading the training output:**
    The console will log messages like:
    ```
    [timestamp] Starting training for 5 epochs. Models will be saved to 'runs/demo' train.py:line
    [timestamp] Epoch 1/5: L_scalar=1.388 | L_gb=0.499                             train.py:line
    ...
    [timestamp] Models saved to runs/demo                                          train.py:line
    ```
    *   `L_scalar`: The sum of BCE losses for the G and B components of the Dual Scalar model. A value around 1.386 (2 * 0.693) suggests the model is predicting ~0.5 for both components (random guessing for binary targets).
    *   `L_gb`: The custom paraconsistent loss for the G/B model.
    Watch for these loss values to decrease, indicating learning.

### 3.2 Evaluating the Models

The `eval.py` script loads the trained models and evaluates them on a new test set.

*   **To run evaluation:**
    ```bash
    python eval.py
    ```
    This assumes models are saved in `runs/demo/`. If you used a different save path during training, you'll need to modify `MODEL_DIR` in `eval.py` accordingly.

*   **What it does:**
    *   Loads the `scalar_model.pth` and `gb_model.pth` from `MODEL_DIR`.
    *   Generates a new test dataset.
    *   Calculates the Root Mean Squared Error (RMSE) for both models.
    *   Displays two plots comparing predictions to targets for the G/B model.

*   **Reading the evaluation output:**
    *   **Console Output:**
        ```
        Loaded trained models from runs/demo
        RMSE Dual Scalar Model: 0.xxxx
        RMSE G/B Model:         0.xxxx
        ```
        RMSE measures the average difference between predicted and target values. Lower is better.
    *   **Plots:**
        1.  **G/B Model Predictions vs Targets:** Shows predicted G vs target G, and predicted B vs target B. Points close to the diagonal `y=x` line indicate better predictions.
        2.  **(G,B) Space: Targets and G/B Model Predictions:** Shows the distribution of (G,B) pairs for targets and the G/B model's predictions in a 2D plot. This helps visualize if the model captures the overall structure of the (G,B) space.

## 4. Next Steps & Experimentation

*   Train for more epochs (`python train.py --epochs N`).
*   Adjust learning rate (`lr`) or G/B loss lambda (`gb_loss_lambda`) in `experiments.cfg`.
*   Modify data generation parameters (`N_ATOMS`, `CONTR_RATE`, `LABEL_NOISE`) in `data/generator.py`.
*   Implement alternative loss functions or model architectures in `models/gb_model.py`.
*   Use the `notebooks/01_quickstart.ipynb` for interactive exploration.

---
This project is based on the plan provided by the user and concepts from paraconsistent logic, particularly the idea of representing propositions with independent "Good" and "Bad" values.
