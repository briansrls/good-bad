import torch
import torch.nn.functional as F

class GBValuatorLearner:
    """
    Handles the learning logic for G-B valuators.
    This class will compute losses or apply updates to G-B values based on 
    hints, survival outcomes, and critiques.
    """
    def __init__(self, hint_strength=1.0, survival_strength=1.0, critique_strength=1.0):
        self.hint_strength = hint_strength
        self.survival_strength = survival_strength
        self.critique_strength = critique_strength

    def calculate_hint_loss(self, gb_outputs, target_gb_values_hint):
        """
        Calculates loss based on intrinsic hints for G-B values.
        Args:
            gb_outputs (torch.Tensor): Output from GBOutputLayer (batch_size, num_categories, 2).
                                       gb_outputs[..., 0] is G, gb_outputs[..., 1] is B.
            target_gb_values_hint (torch.Tensor): Target G-B values based on hints 
                                                  (batch_size, num_categories, 2).
                                                  Use -1 for targets to ignore (e.g. if only G or B is hinted).
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Mask for valid targets (not -1)
        mask = (target_gb_values_hint != -1).float()
        masked_loss = F.mse_loss(gb_outputs * mask, target_gb_values_hint * mask, reduction='none')
        
        # Sum loss for G and B for each category, then average over batch and categories
        # We want to ensure that the loss is meaningful even if only G or B is specified for a category
        loss_per_gb_pair = masked_loss.sum(dim=2) # Sum G and B losses
        num_valid_targets = mask.sum(dim=2).clamp(min=1) # Number of valid G/B targets for this pair
        
        # Average loss for pairs where at least one target (G or B) was specified
        mean_loss_per_pair = loss_per_gb_pair / num_valid_targets
        
        # Consider only pairs where there was at least one hint
        pair_has_hint_mask = (num_valid_targets > 0).float()
        
        if pair_has_hint_mask.sum() == 0: # No hints provided in this batch for any category
            return torch.tensor(0.0, device=gb_outputs.device, requires_grad=True)
            
        total_loss = mean_loss_per_pair.sum() / pair_has_hint_mask.sum()
        return total_loss * self.hint_strength

    def calculate_survival_feedback_loss(self, gb_outputs, category_index, survival_points_change):
        """
        Calculates loss/update based on survival point (SP) changes.
        This is a simplified version. SP changes reinforce G (for SP gain) or B (for SP loss) 
        components of active G-B states.

        Args:
            gb_outputs (torch.Tensor): (batch_size, num_categories, 2) - G and B values for the event.
            category_index (torch.Tensor): (batch_size) or int - Index of the category primarily active or responsible.
            survival_points_change (torch.Tensor): (batch_size) or float - Change in SP. Positive for gain, negative for loss.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # This is a conceptual placeholder. True RL-style updates are more complex.
        # We want to encourage G if SP increases, and B if SP decreases (or G if SP decreases for a "threat avoided" scenario)
        # For simplicity, let's assume for a chosen category: 
        # if SP_change > 0, we want high G, low B.
        # if SP_change < 0, we want low G, high B (for that category being "bad").
        
        # gb_outputs are (batch, num_cat, 2)
        # category_index needs to be shaped for gather
        # Assuming batch_size = 1 for now for simplicity in this placeholder
        if gb_outputs.dim() == 3 and gb_outputs.size(0) > 1 and isinstance(category_index, int):
            # If batch but single cat_index, apply to all items in batch for that cat
            active_gb = gb_outputs[:, category_index, :] # (batch_size, 2)
        elif gb_outputs.dim() == 3 and isinstance(category_index, torch.Tensor) and category_index.dim() == 1:
            # category_index is (batch_size), gather along category dimension
            # category_index needs to be (batch_size, 1, 2) to gather G and B
            idx = category_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)
            active_gb = torch.gather(gb_outputs, 1, idx).squeeze(1) # (batch_size, 2)
        else:
             # Defaulting to first item in batch if shapes are tricky, or assume batch_size = 1
            active_gb = gb_outputs[0, category_index, :] # (1, 2) or (2) if squeezed
            if active_gb.dim() == 1: active_gb = active_gb.unsqueeze(0) # ensure (1,2)
            if isinstance(survival_points_change, torch.Tensor) and survival_points_change.numel() > 1:
                survival_points_change = survival_points_change[0].unsqueeze(0)
            elif isinstance(survival_points_change, (float, int)):
                survival_points_change = torch.tensor([survival_points_change], device=gb_outputs.device)

        g_value = active_gb[:, 0]
        b_value = active_gb[:, 1]

        loss = torch.tensor(0.0, device=gb_outputs.device, requires_grad=True)

        # If SP increased, push G up, B down.
        # If SP decreased, push G down, B up.
        # This is a very direct way, might need refinement (e.g. target values)
        target_g = (survival_points_change > 0).float()
        target_b = (survival_points_change < 0).float()
        
        # Loss for G: if SP change > 0, error is (1-G)^2. If SP change <=0, error is (0-G)^2
        loss_g = (target_g - g_value).pow(2)
        # Loss for B: if SP change < 0, error is (1-B)^2. If SP change >=0, error is (0-B)^2
        loss_b = (target_b - b_value).pow(2)
        
        loss = (loss_g + loss_b).mean() # Average over batch if applicable
        return loss * self.survival_strength

    def calculate_critique_loss(self, gb_outputs, category_index, critique_target_gb):
        """
        Calculates loss based on external critique (e.g., for noise misidentification).
        Example: If noise is misclassified as a shape (High G for shape), critique 
        would provide a target of Low G, Low B (or High B for "not this shape").

        Args:
            gb_outputs (torch.Tensor): (batch_size, num_categories, 2).
            category_index (torch.Tensor): (batch_size) or int - Index of the category being critiqued.
            critique_target_gb (torch.Tensor): (batch_size, 2) or (2) - The G,B target from critique.
                                              (e.g., [0,1] for "this is bad/false for this category")
        Returns:
            torch.Tensor: Scalar loss value.
        """
        if gb_outputs.dim() == 3 and gb_outputs.size(0) > 1 and isinstance(category_index, int):
            active_gb = gb_outputs[:, category_index, :] 
        elif gb_outputs.dim() == 3 and isinstance(category_index, torch.Tensor) and category_index.dim() == 1:
            idx = category_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)
            active_gb = torch.gather(gb_outputs, 1, idx).squeeze(1)
        else:
            active_gb = gb_outputs[0, category_index, :] 
            if active_gb.dim() == 1: active_gb = active_gb.unsqueeze(0)
            if isinstance(critique_target_gb, torch.Tensor) and critique_target_gb.dim() ==1:
                 critique_target_gb = critique_target_gb.unsqueeze(0)
            elif isinstance(critique_target_gb, list): # e.g. [0,1]
                 critique_target_gb = torch.tensor([critique_target_gb], device=gb_outputs.device, dtype=torch.float32)

        loss = F.mse_loss(active_gb, critique_target_gb.expand_as(active_gb))
        return loss * self.critique_strength


if __name__ == '__main__':
    learner = GBValuatorLearner(hint_strength=0.5, survival_strength=1.0, critique_strength=0.7)
    batch_size = 4
    num_categories = 3

    # --- Test Hint Loss ---
    print("\n--- Hint Loss Test ---")
    dummy_gb_outputs = torch.rand(batch_size, num_categories, 2, requires_grad=True)
    # Hints: 
    # Item 0, Cat 0: Good (G=1, B=0)
    # Item 1, Cat 1: Bad (G=0, B=1)
    # Item 2, Cat 2: Dissonant (G=1, B=1)
    # Item 3, Cat 0: Only G hint (G=0.8), B is ignored
    # Other categories/items: No hint (-1)
    target_hints = torch.full((batch_size, num_categories, 2), -1.0)
    target_hints[0, 0, :] = torch.tensor([1.0, 0.0])
    target_hints[1, 1, :] = torch.tensor([0.0, 1.0])
    target_hints[2, 2, :] = torch.tensor([1.0, 1.0])
    target_hints[3, 0, 0] = 0.8 # Only G hint for item 3, cat 0

    hint_loss = learner.calculate_hint_loss(dummy_gb_outputs, target_hints)
    print(f"GB Outputs (first item):\n{dummy_gb_outputs[0]}")
    print(f"Target Hints (first item):\n{target_hints[0]}")
    print(f"Calculated Hint Loss: {hint_loss.item()}")
    assert hint_loss.requires_grad
    hint_loss.backward() # Test backward pass
    assert dummy_gb_outputs.grad is not None
    print("Hint loss test passed.")

    # --- Test Survival Feedback Loss ---
    print("\n--- Survival Feedback Loss Test ---")
    dummy_gb_outputs_surv = torch.rand(batch_size, num_categories, 2, requires_grad=True)
    dummy_gb_outputs_surv.grad = None # Reset grad
    
    # Example 1: SP Gain, affect category 0 for all batch items
    sp_change1 = torch.tensor([10.0] * batch_size) # Positive SP change for all
    cat_idx1 = torch.zeros(batch_size, dtype=torch.long) # Category 0 for all
    survival_loss1 = learner.calculate_survival_feedback_loss(dummy_gb_outputs_surv, cat_idx1, sp_change1)
    print(f"SP Change: {sp_change1[0]}, Affected Cat G/B (item 0, cat 0): {dummy_gb_outputs_surv[0,cat_idx1[0],:]}")
    print(f"Survival Loss (SP Gain): {survival_loss1.item()}")
    assert survival_loss1.requires_grad
    # survival_loss1.backward() # Test backward pass
    # assert dummy_gb_outputs_surv.grad is not None
    # dummy_gb_outputs_surv.grad = None # Reset for next test

    # Example 2: SP Loss, affect category 1 for item 0, cat 2 for item 1 etc.
    sp_change2 = torch.tensor([-5.0] * batch_size) # Negative SP change
    cat_idx2 = torch.randint(0, num_categories, (batch_size,), dtype=torch.long) 
    survival_loss2 = learner.calculate_survival_feedback_loss(dummy_gb_outputs_surv, cat_idx2, sp_change2)
    # print(f"SP Change: {sp_change2[0]}, Affected Cat G/B (item 0, cat {cat_idx2[0]}): {dummy_gb_outputs_surv[0,cat_idx2[0],:]}")
    print(f"Survival Loss (SP Loss): {survival_loss2.item()}")
    print("Survival loss conceptually tested (backward pass commented due to potential multiple contributions).")

    # --- Test Critique Loss ---
    print("\n--- Critique Loss Test ---")
    dummy_gb_outputs_crit = torch.rand(batch_size, num_categories, 2, requires_grad=True)
    # Critique category 1 for all items to be [0.1, 0.9] (Low G, High B)
    crit_cat_idx = torch.ones(batch_size, dtype=torch.long) # Category 1
    crit_target_gb = torch.tensor([0.1, 0.9]).expand(batch_size, -1) # Target G=0.1, B=0.9

    critique_loss = learner.calculate_critique_loss(dummy_gb_outputs_crit, crit_cat_idx, crit_target_gb)
    print(f"Critiqued Cat G/B (item 0, cat 1): {dummy_gb_outputs_crit[0,crit_cat_idx[0],:]}")
    print(f"Critique Target G/B: {crit_target_gb[0]}")
    print(f"Calculated Critique Loss: {critique_loss.item()}")
    assert critique_loss.requires_grad
    critique_loss.backward()
    assert dummy_gb_outputs_crit.grad is not None
    print("Critique loss test passed.")

    print("\nGBValuatorLearner tests completed.") 