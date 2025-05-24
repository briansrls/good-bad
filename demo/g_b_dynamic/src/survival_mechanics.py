class SurvivalMechanicsEngine:
    """
    Manages the agent's Survival Points (SP) and related dynamics.
    """
    def __init__(self, initial_sp=100.0, base_decay_per_cycle=0.1, 
                 sp_gain_opportunity=10.0, sp_loss_threat_unhandled=20.0, 
                 sp_gain_threat_handled=1.0, sp_min_penalty_misclass=1.0,
                 sp_gain_basic_competency=0.5, sp_gain_noise_ignored=0.2):
        """
        Args:
            initial_sp (float): Starting Survival Points.
            base_decay_per_cycle (float): SP lost automatically each cycle/step.
            sp_gain_opportunity (float): SP gained from correctly handling an opportunity.
            sp_loss_threat_unhandled (float): SP lost from failing to handle a threat.
            sp_gain_threat_handled (float): SP gained/loss prevented by correctly handling a threat.
            sp_min_penalty_misclass (float): Minimum SP penalty for misclassification during foundational learning.
            sp_gain_basic_competency (float): SP gained for basic correct actions during foundational learning.
            sp_gain_noise_ignored (float): SP gained for correctly ignoring noise.
        """

        self.current_sp = initial_sp
        self.initial_sp = initial_sp
        self.base_decay_per_cycle = base_decay_per_cycle
        
        self.sp_gain_opportunity = sp_gain_opportunity
        self.sp_loss_threat_unhandled = sp_loss_threat_unhandled
        self.sp_gain_threat_handled = sp_gain_threat_handled
        self.sp_min_penalty_misclass = sp_min_penalty_misclass
        self.sp_gain_basic_competency = sp_gain_basic_competency
        self.sp_gain_noise_ignored = sp_gain_noise_ignored
        
        self.is_alive = True
        self.history = [(0, initial_sp)] # (cycle, SP)
        self.current_cycle = 0

    def cycle_update(self):
        """
        Applies baseline decay for a new cycle.
        Returns the SP change due to decay.
        """
        if not self.is_alive:
            return 0.0
        
        decay_amount = -self.base_decay_per_cycle
        self._update_sp(decay_amount, event_type="decay")
        self.current_cycle += 1
        return decay_amount

    def _update_sp(self, change, event_type="generic_event"):
        """
        Internal method to update SP and check for death condition.
        Args:
            change (float): The amount to change SP by (can be positive or negative).
            event_type (str): Description of the event causing SP change (for logging).
        """
        if not self.is_alive:
            return

        self.current_sp += change
        # print(f"Cycle {self.current_cycle}: SP changed by {change:.2f} due to {event_type}. New SP: {self.current_sp:.2f}")

        if self.current_sp <= 0:
            self.current_sp = 0
            self.is_alive = False
            # print(f"Cycle {self.current_cycle}: Agent has died. SP reached 0.")
        
        self.history.append((self.current_cycle, self.current_sp))

    def record_event(self, event_type, success=True, details=None):
        """
        Records a significant event and updates SP accordingly.

        Args:
            event_type (str): Type of event, e.g., 'foundational_task', 'opportunity', 'threat', 'noise_interaction'.
            success (bool): Whether the event was handled successfully.
            details (dict, optional): Additional details about the event, e.g., {'stimulus_type': 'circle'}.
        
        Returns:
            float: The change in SP due to this event.
        """
        if not self.is_alive:
            return 0.0

        sp_change = 0.0

        if event_type == 'foundational_task': # Covers basic shapes, edges
            if success:
                sp_change = self.sp_gain_basic_competency
            else:
                sp_change = -self.sp_min_penalty_misclass 
        elif event_type == 'noise_interaction':
            if success: # Successfully ignored or correctly identified as noise
                sp_change = self.sp_gain_noise_ignored
            else: # Misclassified noise as signal
                sp_change = -self.sp_min_penalty_misclass 
        elif event_type == 'opportunity':
            if success:
                sp_change = self.sp_gain_opportunity
            else: # Failed to capitalize on opportunity (e.g. misclassified)
                sp_change = -self.sp_min_penalty_misclass # Or a specific penalty for missed opportunity
        elif event_type == 'threat':
            if success: # Correctly identified/handled threat
                sp_change = self.sp_gain_threat_handled
            else: # Failed to identify/handle threat
                sp_change = -self.sp_loss_threat_unhandled
        # Add more event types as needed (e.g., for ambiguous stimuli resolution)
        elif event_type == 'ambiguity_resolved_positively':
             sp_change = self.sp_gain_opportunity # Example: successfully categorizing squircle gives points

        self._update_sp(sp_change, event_type=f"{event_type}_{'success' if success else 'fail'}")
        return sp_change

    def get_sp(self):
        return self.current_sp

    def alive(self):
        return self.is_alive

    def reset(self):
        self.current_sp = self.initial_sp
        self.is_alive = True
        self.current_cycle = 0
        self.history = [(0, self.initial_sp)]

if __name__ == '__main__':
    engine = SurvivalMechanicsEngine(initial_sp=50, base_decay_per_cycle=0.05)
    print(f"Initial SP: {engine.get_sp()}, Alive: {engine.alive()}")

    # Simulate a few cycles with events
    for cycle in range(10):
        decay_sp = engine.cycle_update()
        print(f"Cycle {engine.current_cycle-1}: Decayed by {decay_sp:.2f}. Current SP: {engine.get_sp():.2f}, Alive: {engine.alive()}")
        if not engine.alive(): break

        if cycle % 3 == 0:
            change = engine.record_event('foundational_task', success=True)
            print(f"  Event: Foundational task success. SP Change: {change:.2f}. New SP: {engine.get_sp():.2f}")
        if cycle % 4 == 1:
            change = engine.record_event('threat', success=False)
            print(f"  Event: Threat unhandled. SP Change: {change:.2f}. New SP: {engine.get_sp():.2f}")
            if not engine.alive(): break
        if cycle % 5 == 2:
            change = engine.record_event('opportunity', success=True)
            print(f"  Event: Opportunity success. SP Change: {change:.2f}. New SP: {engine.get_sp():.2f}")
    
    print(f"Final SP: {engine.get_sp():.2f}, Alive: {engine.alive()}")
    print("SP History:", engine.history)

    print("\nResetting engine...")
    engine.reset()
    print(f"SP after reset: {engine.get_sp()}, Alive: {engine.alive()}")

    # Test death condition
    engine = SurvivalMechanicsEngine(initial_sp=10, base_decay_per_cycle=1, sp_loss_threat_unhandled=10)
    print(f"\nTesting death. Initial SP: {engine.get_sp()}")
    engine.cycle_update()
    print(f"SP after 1 cycle: {engine.get_sp()}")
    engine.record_event('threat', success=False)
    print(f"SP after threat: {engine.get_sp()}, Alive: {engine.alive()}")
    assert not engine.alive()
    engine.record_event('opportunity', success=True) # Should have no effect
    print(f"SP after trying event when dead: {engine.get_sp()}")
    assert engine.get_sp() == 0
    print("Death condition test passed.") 