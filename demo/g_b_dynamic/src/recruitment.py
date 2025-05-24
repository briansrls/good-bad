import torch

class RecruitmentManager:
    """
    Manages the dynamic recruitment of L1-E modules based on G-B dissonance.
    """
    def __init__(self, high_g_threshold=0.7, high_b_threshold=0.7, 
                 sustained_duration_threshold=5, # Number of consecutive cycles/timesteps
                 max_l1e_modules=5): # Max L1-E modules to prevent runaway recruitment
        """
        Initializes the recruitment manager.

        Args:
            high_g_threshold (float): Minimum G value to be considered 'High G'.
            high_b_threshold (float): Minimum B value to be considered 'High B'.
            sustained_duration_threshold (int): How many consecutive observations of dissonance 
                                                for a specific category trigger recruitment.
            max_l1e_modules (int): Maximum number of L1-E modules that can be recruited.
        """
        self.high_g_threshold = high_g_threshold
        self.high_b_threshold = high_b_threshold
        self.sustained_duration_threshold = sustained_duration_threshold
        self.max_l1e_modules = max_l1e_modules

        # Tracks dissonance for each category of the L1 network (or potentially an L1-E)
        # Key: category_index, Value: consecutive_dissonance_count
        self.dissonance_tracker = {}
        self.recruited_l1e_count = 0
        
        # Stores info about inputs that triggered dissonance for a category, 
        # which could be useful for initializing/training the new L1-E module.
        # Key: category_index, Value: list of [problematic_input_sample, gb_values]
        self.dissonance_trigger_info = {}

    def check_dissonance_and_trigger(self, gb_outputs, original_inputs=None):
        """
        Checks G-B outputs for dissonance and determines if recruitment should be triggered.

        Args:
            gb_outputs (torch.Tensor): G-B values from a network (batch_size, num_categories, 2).
                                       Assumes gb_outputs[..., 0] is G, gb_outputs[..., 1] is B.
            original_inputs (torch.Tensor, optional): The input stimuli that produced these gb_outputs.
                                                      Shape (batch_size, ...). Stored if dissonance occurs.

        Returns:
            list: A list of category indices for which L1-E recruitment is triggered in this step.
                  Returns an empty list if no recruitment is triggered.
        """
        if self.recruited_l1e_count >= self.max_l1e_modules:
            return [] # Max L1-E modules reached

        batch_size, num_categories, _ = gb_outputs.shape
        triggered_recruitment_for_categories = []

        for i in range(batch_size): # Process each item in the batch
            # Store most recent G/B for each category if we want to reset count on non-dissonance
            current_item_categories_in_dissonance = set()

            for cat_idx in range(num_categories):
                g_value = gb_outputs[i, cat_idx, 0].item()
                b_value = gb_outputs[i, cat_idx, 1].item()

                is_dissonant = (g_value >= self.high_g_threshold and 
                                b_value >= self.high_b_threshold)

                if is_dissonant:
                    current_item_categories_in_dissonance.add(cat_idx)
                    self.dissonance_tracker[cat_idx] = self.dissonance_tracker.get(cat_idx, 0) + 1
                    
                    # Store problematic input and G-B values
                    if original_inputs is not None:
                        if cat_idx not in self.dissonance_trigger_info:
                            self.dissonance_trigger_info[cat_idx] = []
                        # Store a sample (e.g. first one that triggers sustained dissonance)
                        # Or collect a few. For now, just one recent one.
                        self.dissonance_trigger_info[cat_idx].append(
                            (original_inputs[i].clone(), gb_outputs[i, cat_idx, :].clone())
                        )
                        # Keep only a few recent trigger examples per category
                        max_trigger_examples = 5 
                        if len(self.dissonance_trigger_info[cat_idx]) > max_trigger_examples:
                           self.dissonance_trigger_info[cat_idx] = self.dissonance_trigger_info[cat_idx][-max_trigger_examples:]

                    if self.dissonance_tracker[cat_idx] >= self.sustained_duration_threshold:
                        if cat_idx not in triggered_recruitment_for_categories:
                             # Check if we haven't already decided to recruit for this cat_idx due to another batch item
                            triggered_recruitment_for_categories.append(cat_idx)
                            print(f"Recruitment triggered for category {cat_idx} due to sustained dissonance.")
                            # Reset count for this category to prevent immediate re-triggering
                            self.dissonance_tracker[cat_idx] = 0 
                            # self.recruited_l1e_count += 1 # Increment when L1-E is actually created
                else:
                    # If not dissonant, reset the counter for this category for this item
                    # This means sustained means *uninterrupted* dissonance for that category.
                    if cat_idx in self.dissonance_tracker: # only if it was being tracked
                         self.dissonance_tracker[cat_idx] = 0
        
        # If recruitment is triggered for any category, it implies one L1-E module for now.
        # The logic of how many L1-E modules are created and how they specialize will be complex.
        # For now, if list is non-empty, training orchestrator will handle one recruitment.
        if triggered_recruitment_for_categories:
            # self.recruited_l1e_count += 1 # This should be done by the orchestrator when module is created
            pass # Orchestrator will check this list
            
        return triggered_recruitment_for_categories
    
    def get_problematic_stimuli_for_category(self, category_idx):
        """ Returns a list of (input_sample, gb_values) that triggered dissonance for the category. """
        return self.dissonance_trigger_info.get(category_idx, [])

    def confirm_recruitment(self):
        """ Called by orchestrator to confirm an L1-E module has been created. """
        if self.recruited_l1e_count < self.max_l1e_modules:
            self.recruited_l1e_count += 1
            return True
        return False
        
    def reset(self):
        self.dissonance_tracker = {}
        self.dissonance_trigger_info = {}
        self.recruited_l1e_count = 0

if __name__ == '__main__':
    manager = RecruitmentManager(high_g_threshold=0.7, high_b_threshold=0.7, sustained_duration_threshold=3)
    batch_s = 2
    num_cats = 2

    print("--- Recruitment Manager Test ---")

    # Scenario 1: No dissonance
    gb_low = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.2,0.1],[0.1,0.1]]]) # (2,2,2)
    triggered = manager.check_dissonance_and_trigger(gb_low)
    print(f"Low Dissonance: Triggered cats: {triggered}, Dissonance Counts: {manager.dissonance_tracker}")
    assert not triggered

    # Scenario 2: Dissonance for cat 0, but not sustained
    gb_dissonant_cat0_once = torch.tensor([[[0.8, 0.8], [0.1, 0.1]], [[0.8,0.9],[0.1,0.1]]])
    dummy_inputs = torch.randn(batch_s, 1, 8, 8) # Dummy inputs for storage

    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs)
    print(f"Dissonance Cat 0 (1st time): Triggered cats: {triggered}, Dissonance Counts: {manager.dissonance_tracker}")
    assert not triggered
    assert manager.dissonance_tracker.get(0,0) == batch_s # Both items in batch showed dissonance for cat 0

    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs)
    print(f"Dissonance Cat 0 (2nd time): Triggered cats: {triggered}, Dissonance Counts: {manager.dissonance_tracker}")
    assert not triggered
    assert manager.dissonance_tracker.get(0,0) == batch_s * 2

    # Scenario 3: Sustained dissonance for cat 0 triggers recruitment
    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs)
    print(f"Dissonance Cat 0 (3rd time): Triggered cats: {triggered}, Dissonance Counts: {manager.dissonance_tracker}")
    assert 0 in triggered
    assert manager.dissonance_tracker.get(0,0) == 0 # Reset after trigger for that category
    # manager.confirm_recruitment() # Orchestrator would call this
    # print(f"Recruited L1-E count: {manager.recruited_l1e_count}")

    # Scenario 4: Dissonance for cat 1, then stops, then cat 0 again
    manager.reset()
    # print(f"Recruited L1-E count after reset: {manager.recruited_l1e_count}")
    gb_dissonant_cat1 = torch.tensor([[[0.1, 0.1], [0.9, 0.9]], [[0.1,0.1],[0.8,0.8]]])
    manager.check_dissonance_and_trigger(gb_dissonant_cat1, dummy_inputs) # Cat 1 dissonant (count 2 due to batch)
    print(f"Dissonance Cat 1 (1st time): Counts {manager.dissonance_tracker}") 
    manager.check_dissonance_and_trigger(gb_low, dummy_inputs) # No dissonance, cat 1 count resets
    print(f"No Dissonance: Counts {manager.dissonance_tracker}")
    assert manager.dissonance_tracker.get(1,0) == 0
    
    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs) # Cat 0 dissonant (count 2)
    print(f"Dissonance Cat 0 (1st time again): Counts {manager.dissonance_tracker}")
    assert manager.dissonance_tracker.get(0,0) == 2
    assert not triggered

    # Scenario 5: Max L1-E modules
    manager.reset()
    manager.max_l1e_modules = 1
    manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs) # count=2
    manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs) # count=4
    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs) # count=6, triggers
    assert 0 in triggered
    manager.confirm_recruitment() # Manually confirm for test
    print(f"Recruited L1-E count: {manager.recruited_l1e_count}")
    
    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs) # count=2
    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs) # count=4
    triggered = manager.check_dissonance_and_trigger(gb_dissonant_cat0_once, dummy_inputs) # count=6, should trigger cat 0 again but max reached
    print(f"Trying to trigger again: Triggered: {triggered}")
    assert not triggered # No new trigger as max_l1e_modules reached
    print("Recruitment manager tests passed.") 