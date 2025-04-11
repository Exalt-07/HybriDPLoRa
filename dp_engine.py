# Define Adaptive Differential Privacy Engine
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

class AdaptiveDPE:
    """
    Adaptive Differential Privacy Engine for HybridDP-LoRA
    Dynamically adjusts noise during federated training
    """
    def __init__(
        self, 
        target_epsilon: float = 4.0, 
        target_delta: float = 1e-5,
        initial_noise_multiplier: float = 1.2,
        max_grad_norm: float = 0.7
    ):
        self.accountant = RDPAccountant()
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = initial_noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.privacy_engine = PrivacyEngine()
        
    def make_private(self, model, optimizer, data_loader=None):
        """Apply DP-SGD to model training"""
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        return model, optimizer, data_loader
        
    def update_noise_levels(self, config):
        """Adaptively adjust noise multiplier based on privacy budget"""
        current_epsilon = self.accountant.get_epsilon(self.target_delta)
        
        if current_epsilon > 0.75 * self.target_epsilon:
            # Increase noise if approaching privacy budget
            self.noise_multiplier *= 1.1
            print(f"Privacy budget threshold reached. Increasing noise to {self.noise_multiplier:.4f}")
        
        # Update configuration for federated clients
        config['noise_multiplier'] = self.noise_multiplier
        config['max_grad_norm'] = self.max_grad_norm
        return config
    
    def get_privacy_spent(self):
        """Report current privacy guarantee"""
        epsilon = self.accountant.get_epsilon(self.target_delta)
        return epsilon, self.target_delta
