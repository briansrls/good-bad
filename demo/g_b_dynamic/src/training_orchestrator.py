import torch
import torch.optim as optim
import numpy as np
import os

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from .stimulus_generator import generate_simple_shape, generate_noise_field, generate_line, generate_squircle, generate_star, generate_hexagon, generate_threat_shape
from .neural_core import L1Network, L1ENetwork # GBOutputLayer is used internally by these
from .gb_valuator import GBValuatorLearner
from .survival_mechanics import SurvivalMechanicsEngine
from .recruitment import RecruitmentManager
from .visualization import plot_full_experiment_visualization

class TrainingOrchestrator:
    """
    Orchestrates the training process, including L1 foundational learning,
    L1-E module recruitment, and overall survival challenge progression.
    """
    def __init__(self, config):
        """
        Initializes the orchestrator with experiment configurations.
        Args:
            config (dict): A dictionary containing configuration parameters such as:
                image_size, input_channels, num_l1_categories, l1_fc_hidden_features,
                initial_sp, learning_rate, batch_size, num_cycles_phase1, num_cycles_phase2,
                recruitment_thresholds (g, b, duration), etc.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize components
        self.l1_network = L1Network(
            input_channels=config.get('input_channels', 1),
            image_size=config.get('image_size', 32),
            num_categories=config.get('num_l1_categories', 3), # e.g., shape1, shape2, noise
            fc_hidden_features=config.get('l1_fc_hidden_features', 128)
        ).to(self.device)

        self.l1e_networks = [] # List to hold L1-E modules
        self.active_network = self.l1_network # Initially, only L1 is active

        self.optimizer_l1 = optim.Adam(self.l1_network.parameters(), lr=config.get('learning_rate_l1', 0.001))
        self.optimizers_l1e = []

        self.gb_learner = GBValuatorLearner(
            hint_strength=config.get('hint_strength', 1.0),
            survival_strength=config.get('survival_strength', 1.0),
            critique_strength=config.get('critique_strength', 1.0)
        )
        self.survival_engine = SurvivalMechanicsEngine(
            initial_sp=config.get('initial_sp', 100.0),
            base_decay_per_cycle=config.get('base_decay_per_cycle', 0.1)
            # Other SP parameters can be added to config
        )
        self.recruitment_manager = RecruitmentManager(
            high_g_threshold=config.get('rec_high_g', 0.7),
            high_b_threshold=config.get('rec_high_b', 0.7),
            sustained_duration_threshold=config.get('rec_duration', 5),
            max_l1e_modules=config.get('max_l1e_modules', 5)
        )
        
        # Define categories (example, should be configurable or derived)
        # These need to map to indices used by the network output and learner
        self.l1_category_map = config.get('l1_category_map', {'circle': 0, 'square': 1, 'noise': 2})
        # L1-E categories will be defined when they are recruited

        self.current_phase = 0 # 0: Pre-training, 1: Phase 1, 2: Phase 2
        self.current_cycle = 0

        self.phase1_loss_log = []
        self.phase2_loss_log = []
        
        # Initialize probe stimuli and log for G-B evolution
        self._initialize_probe_stimuli()
        # Structure: self.probe_gb_log[probe_name][category_name]['g'] = [(cycle, value), ...]
        # Structure: self.probe_gb_log[probe_name][category_name]['b'] = [(cycle, value), ...]
        self.probe_gb_log = {name: {cat_name: {'g': [], 'b': []} for cat_name in self.l1_category_map.keys()} 
                             for name in self.probe_stimuli.keys()}

    def _initialize_probe_stimuli(self):
        """Generates and stores a fixed set of probe stimuli."""
        img_size = (self.config.get('image_size', 32), self.config.get('image_size', 32))
        # Using very low noise for "clean" probes
        probe_noise = self.config.get('probe_stimulus_noise_level', 0.01) 
        
        self.probe_stimuli = {}
        self.probe_stimuli['probe_circle'] = torch.from_numpy(
            generate_simple_shape(image_size=img_size, shape_type='circle', noise_level=probe_noise)
        ).float().unsqueeze(0).unsqueeze(0).to(self.device) # Add batch and channel dim

        self.probe_stimuli['probe_square'] = torch.from_numpy(
            generate_simple_shape(image_size=img_size, shape_type='square', noise_level=probe_noise)
        ).float().unsqueeze(0).unsqueeze(0).to(self.device)

        self.probe_stimuli['probe_noise'] = torch.from_numpy(
            generate_noise_field(image_size=img_size, noise_intensity=0.5) # Intensity for noise field
        ).float().unsqueeze(0).unsqueeze(0).to(self.device)
        print(f"Initialized {len(self.probe_stimuli)} probe stimuli.")

    def _log_probe_gb_values(self):
        """Logs G-B values from L1 network for predefined probe stimuli."""
        if not hasattr(self, 'probe_stimuli') or not self.probe_stimuli:
            return

        # print(f"DEBUG: Logging probe G-B values for cycle {self.current_cycle}") # Optional: overall debug for this function call
        original_mode_is_train = self.l1_network.training
        self.l1_network.eval() # Ensure network is in evaluation mode for probing
        with torch.no_grad(): # No need to track gradients for probing
            for probe_name, stimulus_tensor in self.probe_stimuli.items():
                tensor_sample_sum = stimulus_tensor.sum().item()
                gb_outputs = self.l1_network(stimulus_tensor) # Shape (1, num_categories, 2)
                
                print(f"  DEBUG Cycle {self.current_cycle} - Probe: {probe_name}, Input Sum: {tensor_sample_sum:.4f}")
                print(f"    L1 Output G/B:")
                
                # Create a reverse map from cat_idx to cat_name for easier lookup if multiple names map to one index
                idx_to_name_map = {idx: [] for idx in range(gb_outputs.shape[1])}
                for name, idx in self.l1_category_map.items():
                    if 0 <= idx < gb_outputs.shape[1]:
                        idx_to_name_map[idx].append(name)
                
                for cat_idx in range(gb_outputs.shape[1]):
                    cat_names_for_idx = idx_to_name_map.get(cat_idx, [f"UnknownCat{cat_idx}"])
                    # Join multiple names if an index is shared (e.g. circle/line for cat 0)
                    display_cat_name = "/".join(cat_names_for_idx)
                    
                    g_val = gb_outputs[0, cat_idx, 0].item()
                    b_val = gb_outputs[0, cat_idx, 1].item()
                    print(f"      {display_cat_name} (Cat {cat_idx}): G={g_val:.4f}, B={b_val:.4f}")
                    
                    # Logging to self.probe_gb_log remains the same, using primary cat_name from l1_category_map iteration
                    # This part is a bit tricky if multiple names map to the same cat_idx for logging structure.
                    # The current logging iterates l1_category_map, so it uses the specific names from there.
                    # This debug print aims to show all network outputs clearly.

                # Original logging loop (ensure it still works as intended with the new print format above)
                for cat_name_from_map, cat_idx_from_map in self.l1_category_map.items():
                    # This ensures we log based on the primary names in l1_category_map for consistency with plotting
                    if 0 <= cat_idx_from_map < gb_outputs.shape[1]: 
                        g_val_log = gb_outputs[0, cat_idx_from_map, 0].item()
                        b_val_log = gb_outputs[0, cat_idx_from_map, 1].item()
                        
                        # Check if probe_name and cat_name_from_map path exists in log, if not, something is wrong with init
                        if probe_name in self.probe_gb_log and cat_name_from_map in self.probe_gb_log[probe_name]:
                            self.probe_gb_log[probe_name][cat_name_from_map]['g'].append((self.current_cycle, g_val_log))
                            self.probe_gb_log[probe_name][cat_name_from_map]['b'].append((self.current_cycle, b_val_log))
                        # else: print(f"Warning: Could not log for {probe_name} - {cat_name_from_map}") # Should not happen
        
        if original_mode_is_train:
            self.l1_network.train() # Revert to original training mode

    def _get_stimulus_and_hint_phase1(self, batch_size):
        """ Generates a batch of stimuli and corresponding G-B hints for Phase 1. """
        images = []
        target_gbs = [] # Target G-B values for hints
        event_types = []
        img_size = (self.config.get('image_size', 32), self.config.get('image_size', 32))
        num_l1_cats = self.config.get('num_l1_categories', 3)

        for _ in range(batch_size):
            stim_type_key = np.random.choice(list(self.l1_category_map.keys())) # Choose from defined L1 keys, e.g. 'circle', 'square', 'noise' (if 'line' is in map, it can be chosen too)
            # Ensure 'line' is treated as one of the primary stimulus types if it maps to a base category like circle or is its own category
            # For simplicity, let's assume self.l1_category_map keys are the actual stimuli we can generate directly for Phase 1 basic learning
            # e.g. if l1_category_map = {'circle':0, 'square':1, 'noise':2}, then stim_type_key will be one of these.

            noise = self.config.get('stimulus_noise_level', 0.1)
            
            # Initialize hints: G=0.0, B=0.1 for all categories (default negative hint)
            current_target_gb = torch.full((num_l1_cats, 2), 0.0)
            current_target_gb[:, 1] = 0.1 # B=0.1 for all

            event_type = "foundational_task"
            img = None

            # Determine the actual stimulus to generate based on stim_type_key
            # And set the positive hint for the target category
            target_cat_idx = self.l1_category_map.get(stim_type_key)

            if target_cat_idx is None: # Should not happen if stim_type_key is from l1_category_map keys
                print(f"Warning: stim_type_key '{stim_type_key}' not in l1_category_map. Generating noise.")
                img = generate_noise_field(image_size=img_size, noise_intensity=0.7)
                stim_type_key = 'noise' # Update for event type logging
                target_cat_idx = self.l1_category_map.get('noise', num_l1_cats -1) # Default to last cat for noise if problematic
                event_type = "noise_interaction"
                # For noise, the hint should be low G, low B. Override default negative for other cats.
                current_target_gb = torch.full((num_l1_cats, 2), 0.1) # G=0.1, B=0.1 for all if input is actual noise
                if 0 <= target_cat_idx < num_l1_cats: # Explicitly set noise cat hint if defined
                    current_target_gb[target_cat_idx, :] = torch.tensor([0.1, 0.1]) 
            
            elif stim_type_key == 'circle':
                img = generate_simple_shape(image_size=img_size, shape_type='circle', noise_level=noise)
                current_target_gb[target_cat_idx, :] = torch.tensor([1.0, 0.0]) # Positive hint
            elif stim_type_key == 'square':
                img = generate_simple_shape(image_size=img_size, shape_type='square', noise_level=noise)
                current_target_gb[target_cat_idx, :] = torch.tensor([1.0, 0.0]) # Positive hint
            elif stim_type_key == 'noise': # This refers to the category 'noise', distinct from general noise field for fallback
                img = generate_noise_field(image_size=img_size, noise_intensity=0.7)
                event_type = "noise_interaction"
                # For actual noise input, all categories should have low G, low B (or neutral)
                current_target_gb = torch.full((num_l1_cats, 2), 0.1) # G=0.1, B=0.1 for all
                current_target_gb[target_cat_idx, :] = torch.tensor([0.1, 0.1]) # Explicitly target noise category hint
            elif stim_type_key == 'line': # If 'line' is a distinct key in l1_category_map
                 # Assuming 'line' maps to e.g. category 0 (like circle) as per current DEFAULT_CONFIG
                img = generate_line(image_size=img_size, orientation=np.random.choice(['horizontal','vertical']), noise_level=noise)
                # The target_cat_idx for 'line' (e.g., 0) will get G=1,B=0
                # Other categories (e.g., 1 'square', 2 'noise') will keep G=0,B=0.1
                current_target_gb[target_cat_idx, :] = torch.tensor([1.0, 0.0]) 
            else: # Fallback for other potential keys if l1_category_map is extended
                print(f"Warning: Unhandled stim_type_key '{stim_type_key}' in Phase 1. Generating noise.")
                img = generate_noise_field(image_size=img_size, noise_intensity=0.7)
                event_type = "noise_interaction"
                # Neutral hint for all categories if input is unknown
                current_target_gb = torch.full((num_l1_cats, 2), 0.1)

            if img is None: # Safety net if img wasn't generated
                print(f"Error: Image not generated for stim_type_key '{stim_type_key}'. Using noise.")
                img = generate_noise_field(image_size=img_size, noise_intensity=0.7)
                event_type = "noise_interaction"
                current_target_gb = torch.full((num_l1_cats, 2), 0.1)
            
            images.append(torch.from_numpy(img).float().unsqueeze(0)) # Add channel dim
            target_gbs.append(current_target_gb)
            event_types.append(event_type)
            
        batch_images = torch.stack(images).to(self.device)
        batch_target_gbs = torch.stack(target_gbs).to(self.device)
        return batch_images, batch_target_gbs, event_types

    def run_phase1(self):
        """ Runs Phase 1: Foundational Learning for L1. """
        print("\n--- Starting Phase 1: Foundational Learning ---")
        self.current_phase = 1
        self.l1_network.train()
        
        num_cycles = self.config.get('num_cycles_phase1', 100)
        batch_size = self.config.get('batch_size', 32)

        for cycle in range(num_cycles):
            if not self.survival_engine.alive():
                print(f"Agent died at cycle {self.current_cycle}. Ending Phase 1 early.")
                break
            self.current_cycle +=1
            self.survival_engine.cycle_update() # Apply base decay

            # Get stimuli and hints
            stimuli, target_gbs_hint, event_types_true = self._get_stimulus_and_hint_phase1(batch_size)
            
            # Forward pass (L1 only)
            self.optimizer_l1.zero_grad()
            gb_outputs_l1 = self.l1_network(stimuli)
            
            # Calculate losses
            hint_loss = self.gb_learner.calculate_hint_loss(gb_outputs_l1, target_gbs_hint)
            
            # For Phase 1, SP changes are simpler, tied to hints primarily
            # We can simulate SP changes based on how well hints are matched or basic competency
            # This is a simplification; true SP interaction might be more complex
            batch_sp_change = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                # Simplified: if primary hint target is met, SP gain, else penalty
                # This needs better logic based on which category was targeted by the hint
                # For now, let's assume a generic SP change for the event type
                # The SurvivalMechanicsEngine's record_event gives SP based on 'success'
                # What defines success here? For hints, if the loss is low for that item?
                # Let's use a placeholder: assume success for now and survival_engine handles it.
                # This part needs refinement: how to link hint loss to SP change for *each* item.
                sp_change_item = self.survival_engine.record_event(event_types_true[i], success=True) # Placeholder success
                batch_sp_change[i] = sp_change_item

            # The gb_learner's survival_loss part is more for Phase 2 where SP is direct feedback
            # In Phase 1, SP is simpler and hints are primary. So total_loss is mainly hint_loss.
            total_loss = hint_loss 
            # Add critique loss if applicable (e.g. if noise is misclassified despite hint)
            # Add survival_feedback_loss if SP changes are used directly for G/B gradient

            if total_loss.requires_grad: # Avoid backward on zero loss if no hints were applicable
                total_loss.backward()
                self.optimizer_l1.step()

            if cycle % self.config.get('log_interval', 10) == 0:
                print(f"Phase 1, Cycle {self.current_cycle}/{num_cycles}, L1 Loss: {total_loss.item():.4f}, SP: {self.survival_engine.get_sp():.2f}")
                self.phase1_loss_log.append(total_loss.item())
                # Before logging probes, ensure L1 is in train mode if it should be
                # _log_probe_gb_values will set it to eval and then back to its original mode.
                self.l1_network.train() # Explicitly set to train before calling log, which sets to eval temporarily
                self._log_probe_gb_values() 
        
        print("--- Phase 1 Finished ---")

    def run_phase2(self):
        """ Runs Phase 2: Ambiguity, Novelty, Critical Stimuli, L1-E Recruitment. """
        print("\n--- Starting Phase 2: Adaptation and Survival ---")
        self.current_phase = 2
        self.l1_network.eval() # L1 is mostly frozen or has very slow learning rate
        for l1e_net in self.l1e_networks: l1e_net.train()

        num_cycles = self.config.get('num_cycles_phase2', 200)
        batch_size = self.config.get('batch_size', 16) # Potentially smaller for more targeted L1-E training

        for cycle in range(num_cycles):
            if not self.survival_engine.alive():
                print(f"Agent died at cycle {self.current_cycle}. Ending Phase 2 early.")
                break
            self.current_cycle += 1
            self.survival_engine.cycle_update()

            # TODO: Get stimuli for Phase 2 (mix of known, ambiguous, novel, threats)
            # stimuli, stimulus_info = self._get_stimulus_phase2(batch_size)
            stimuli = None # Placeholder
            if stimuli is None: # Temp: use phase 1 stimuli for testing flow
                 stimuli, _, _ = self._get_stimulus_and_hint_phase1(batch_size)

            # --- Main Agent Forward Pass & Decision Making (L1 + L1Es) ---
            # This part needs careful design: how L1 and L1-Es interact.
            # Possibilities: 
            # 1. L1 processes first. If dissonance, L1-E for that category gets a chance.
            # 2. Gating network / router decides if L1 or an L1-E handles input.
            # 3. L1-E output overrides/modulates L1 for specific stimuli it handles.
            # For now, let L1 always process. If L1-Es exist, they might also process or be chosen.
            
            gb_outputs_l1 = self.l1_network(stimuli)
            final_gb_outputs = gb_outputs_l1 # Default to L1 output
            active_network_for_loss = self.l1_network # Which network parameters to update
            active_optimizer = self.optimizer_l1
            
            # TODO: If L1-Es exist, how do they contribute or take over?
            # This is a placeholder for a more complex integration strategy.
            # For now, let's assume L1-E modules are trained on inputs that L1 found dissonant.

            # --- Recruitment Check (based on L1's output for now) ---
            # In a more complex system, dissonance could arise from L1-E too.
            triggered_categories = self.recruitment_manager.check_dissonance_and_trigger(gb_outputs_l1, stimuli)
            
            if triggered_categories and self.recruitment_manager.confirm_recruitment():
                print(f"Orchestrator: Recruitment confirmed for categories: {triggered_categories}. Creating L1-E.")
                # For now, one L1-E handles the first triggered category. More complex mapping needed.
                # Or one L1-E could try to handle multiple if they are related.
                new_l1e_category_label = f"l1e_specialized_for_l1cat_{triggered_categories[0]}"
                num_l1e_cats = self.config.get('num_l1e_categories', 1) # Each L1-E specializes
                
                new_l1e = L1ENetwork(
                    input_channels=self.config.get('input_channels', 1),
                    image_size=self.config.get('image_size', 32),
                    num_categories=num_l1e_cats, # Specialized categories for this L1-E
                    fc_hidden_features=self.config.get('l1e_fc_hidden_features', 64),
                    l1_context_features=self.config.get('l1e_context_features', 0) # Placeholder
                ).to(self.device)
                self.l1e_networks.append(new_l1e)
                self.optimizers_l1e.append(optim.Adam(new_l1e.parameters(), lr=self.config.get('learning_rate_l1e', 0.001)))
                print(f"  New L1-E module #{len(self.l1e_networks)} created for L1 category {triggered_categories[0]}.")
                # TODO: Initialize/pre-train L1-E with problematic_stimuli from recruitment_manager

            # --- Loss Calculation & Backprop ---
            # This will depend on whether L1 or an L1-E is being trained for this batch.
            # And how SP changes are attributed. For now, apply SP feedback to L1 if no L1-E is active for this.
            # If an L1-E *is* active and handles this stimulus, feedback should go to it.
            
            # Simplified loss for testing flow - just use L1 outputs and dummy SP changes
            dummy_cat_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device) # Assume cat 0
            dummy_sp_change = torch.randn(batch_size, device=self.device) * 5 # Random SP changes
            # Ensure final_gb_outputs does not require grad if L1 is frozen and no L1E is active yet
            # or handle it appropriately if L1E outputs are used for loss.
            # For now, final_gb_outputs is from L1 which is in eval mode.
            survival_loss = self.gb_learner.calculate_survival_feedback_loss(final_gb_outputs.detach(), dummy_cat_idx, dummy_sp_change)
            total_loss = survival_loss # This loss is not used to update L1 if L1 is frozen.
            
            # --- L1 / L1-E Updates ---
            # L1 is in eval() mode, so its weights should not be updated by optimizer_l1
            # If an L1-E module were active and being trained, its specific optimizer would be used here.
            # For example:
            # if active_l1e_network and active_l1e_optimizer:
            #     active_l1e_optimizer.zero_grad()
            #     # L1-E loss would need to be calculated based on L1-E outputs
            #     # l1e_loss = calculate_l1e_loss(l1e_outputs, ...)
            #     if l1e_loss.requires_grad:
            #         l1e_loss.backward()
            #         active_l1e_optimizer.step()
            
            # The previous logic for L1 update was: 
            # if self.l1_network.training: 
            #     self.optimizer_l1.zero_grad()
            #     if total_loss.requires_grad:
            #         total_loss.backward()
            #         self.optimizer_l1.step()
            # Since self.l1_network is in eval() mode for Phase 2, the above block correctly does nothing for L1.
            # The total_loss calculated above is based on L1 outputs, but it is not backpropagated through L1 here.

            if cycle % self.config.get('log_interval', 10) == 0:
                print(f"Phase 2, Cycle {self.current_cycle}/{num_cycles}, Loss: {total_loss.item():.4f}, SP: {self.survival_engine.get_sp():.2f}, L1-Es: {len(self.l1e_networks)}")
                self.phase2_loss_log.append(total_loss.item())
                # _log_probe_gb_values logs L1's response. L1 is already in eval mode for Phase 2.
                # L1-Es are in train mode, but _log_probe_gb_values only uses self.l1_network.
                self._log_probe_gb_values() 
                # No need to change L1 mode here as it should stay eval.
                # Ensure L1-Es are still in train mode if _log_probe_gb_values affected them (it shouldn't).
                # self.l1_network.eval() # Already in eval
                # for l1e_net in self.l1e_networks: l1e_net.train() # Should still be in train

        print("--- Phase 2 Finished ---")

    def run_experiment(self):
        self.run_phase1()
        if self.survival_engine.alive():
            self.run_phase2()
        else:
            print("Agent did not survive Phase 1. Skipping Phase 2.")
        
        print("\n--- Generating Plots ---")
        plot_dir = self.config.get("plot_directory", "plots")
        show_plots_after_run = self.config.get("show_plots_after_run", True)
        
        # Define a single path for the combined plot
        full_plot_filename = self.config.get("full_plot_filename", "full_experiment_visualization.png")
        full_plot_path = os.path.join(plot_dir, full_plot_filename)

        # Call the new comprehensive plotting function
        plot_full_experiment_visualization(
            sp_history=self.survival_engine.history,
            phase1_loss_log=self.phase1_loss_log,
            phase2_loss_log=self.phase2_loss_log,
            probe_gb_log=self.probe_gb_log,
            l1_category_map=self.l1_category_map,
            save_path=full_plot_path,
            show_plot=show_plots_after_run
        )

        print("--- Plotting Complete ---")
        
        print(f"\nExperiment Finished. Final SP: {self.survival_engine.get_sp():.2f}, Alive: {self.survival_engine.alive()}, L1-E modules: {len(self.l1e_networks)}")

# Example Configuration (can be loaded from YAML or defined in main.py)
DEFAULT_CONFIG = {
    'image_size': 32,
    'input_channels': 1,
    'num_l1_categories': 3, # e.g., Circle, Square, Noise
    'l1_category_map': {'circle': 0, 'square': 1, 'noise': 2, 'line':0 }, # line maps to circle for now
    'l1_fc_hidden_features': 64,
    'learning_rate_l1': 0.001,
    'learning_rate_l1e': 0.001,
    'batch_size': 16,
    'num_cycles_phase1': 100,
    'num_cycles_phase2': 100,
    'log_interval': 5,
    'initial_sp': 100.0,
    'base_decay_per_cycle': 0.05,
    'hint_strength': 1.0,
    'survival_strength': 1.5,
    'critique_strength': 1.0,
    'stimulus_noise_level': 0.05,
    'rec_high_g': 0.7,
    'rec_high_b': 0.7,
    'rec_duration': 3, # Shorter for testing
    'max_l1e_modules': 2,
    'num_l1e_categories': 1, # Each L1-E specializes in one new category (conceptually)
    'l1e_fc_hidden_features': 32,
    'l1e_context_features': 0, # Not used yet
    'probe_stimulus_noise_level': 0.01,
    # Updated to a single filename for the full visualization
    'full_plot_filename': "full_experiment_visualization.png"
}

if __name__ == '__main__':
    print("Testing Training Orchestrator...")
    # Use a default config for direct testing
    orchestrator = TrainingOrchestrator(DEFAULT_CONFIG)
    orchestrator.run_experiment()
    print("Training Orchestrator test run completed.") 