# Dynamic Survival Learner

## Project Overview

This project aims to implement the "Dynamic, Survival-Oriented Learning Architecture" described in the design document `demo/g_b_dynamic/dynamic_learning.tex`. The core idea is to create an AI agent that can dynamically adapt its neural architecture in response to cognitive dissonance (conflicting internal Good-Bad signals) when faced with ambiguous or novel stimuli. The agent's learning and adaptation are guided by a "Survival Points" (SP) system, reflecting its ability to achieve goals and maintain coherence.

The architecture starts with a foundational learning layer (L1) and can recruit L1-Expansion (L1-E) modules to handle situations L1 cannot resolve. This process is inspired by the Emotional Optimization Robots (EOR) model.

## Directory Structure

The project is organized as follows:

*   `data/`: Contains modules for generating and managing stimuli.
    *   `stimulus_generator.py`: Responsible for creating 2D image stimuli (e.g., shapes, noise) and associating them with characteristics relevant for G-B evaluation and survival.
*   `core/`: Houses the central components of the learning architecture.
    *   `l1_network.py`: Defines the L1 foundational neural network (likely a CNN) for initial processing and G-B evaluation.
    *   `l1_e_network.py`: Defines the architecture for L1-Expansion (L1-E) modules, which are recruited dynamically.
    *   `gb_valuator.py`: Implements the Good-Bad (G-B) valuator logic, which outputs G (Goodness) and B (Badness) signals. It will likely adapt or utilize `modules/gb_core.py`.
    *   `recruitment_manager.py`: Manages the triggering and instantiation of L1-E modules based on G-B dissonance signals from L1.
*   `modules/`: Contains potentially reusable components, including adaptations from other projects.
    *   `gb_core.py`: Adapted from `demo/g_b_learner/models/gb_model.py`, this provides foundational code for G-B models and loss functions.
*   `survival/`: Implements the agent's survival mechanics.
    *   `sp_engine.py`: Manages the "Survival Points" (SP) system, including SP accumulation, decay, and adjustments based on performance and environmental feedback.
*   `training/`: Contains scripts for training the agent.
    *   `main_trainer.py`: Orchestrates the different training phases for L1 and L1-E modules, integrating all other components.
*   `visualization/`: Tools for visualizing the agent's state and learning process.
    *   `dashboard.py`: Placeholder for a future dashboard to display internal G-B states, SP trajectories, network structure, etc.
*   `experiments/`: For storing experiment configurations, results, and logs.

## Key Unanswered Questions & Future Clarifications

This initial structure lays the groundwork. Further development will need to address:

1.  **L1 CNN Architecture:** Specific details of the L1 CNN (layers, filter sizes, activation functions).
2.  **SP Engine Details:**
    *   Precise mechanics for SP calculation (e.g., impact of different stimuli, decay rates).
    *   How SP changes directly influence the learning of G-B valuators (e.g., reinforcement mechanisms).
3.  **L1-E Module Management:**
    *   Initialization process for new L1-E modules (e.g., random weights, partial copy of L1, specific sub-architecture).
    *   Mechanism for integrating L1-E outputs with L1 (e.g., override, weighted sum, gating).
4.  **Guidance Mechanisms:**
    *   Implementation details for "intrinsic hints" for early L1 learning.
    *   How "external critique" (e.g., for noise misidentification) will be provided and integrated.
5.  **`g_b_learner` Integration Strategy:**
    *   The `modules/gb_core.py` is a direct copy for now. A more refined strategy (refactoring into a library, selective use of functions/classes) might be needed.
    *   The `demo/g_b_learner/data/generator.py` was not directly copied due to different stimulus requirements (one-hot atoms vs. images), but may offer insights for G/B value association.
6.  **Data Formats:**
    *   Standardized format for input stimuli (e.g., tensor dimensions, normalization).
    *   Representation of G-B targets/values during training and inference.
7.  **G-B Dissonance Thresholds:** Specific values and duration for "High G", "High B" that trigger L1-E recruitment.
8.  **L1-E Learning Objectives:** How the L1-E module is specifically tasked with resolving the ambiguity (e.g., loss functions for L1-E).

## Development Approach

The project will likely follow the phases outlined in the design document:
1.  Core Mechanics (L1, Basic G-B, Basic Survival, Phase 1 Stimuli)
2.  Dynamic Recruitment (L1-E instantiation, Ambiguous Shapes)
3.  Full Survival Challenge & Evaluation

This README will be updated as the project progresses.
