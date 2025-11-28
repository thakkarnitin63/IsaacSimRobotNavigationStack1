# In: navigation_stack/controllers/mppi_noise.py
from .mppi_types import MPPIConfig, MPPIState, ControlSequence

import torch
from typing import Tuple

class NoiseGenerator:
    """
    Generates exploration noise for control sampling
    Randomness in mppi
    """
    def __init__(self, config: MPPIConfig):
        self.config = config
        self.device = config.device

        # Validate that noise parameters are reasonable
        if config.v_std <= 0 or config.w_std <= 0:
            raise ValueError("Noise std devs must be positive!")
        
        if config.v_std > config.v_max:
            print(f" WARNING: v_std ({config.v_std}) > v_max ({config.v_max})")
            print("   This means noise is larger than velocity range!")

    def generate_noisy_controls(
        self,
        nominal_sequence: ControlSequence)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (v_samples, w_samples) both shape [samples, horizon]
        """
        K = self.config.num_samples
        T = self.config.horizon_steps

        # Generate noise: [K,T]
        v_noise = torch.randn(K, T, device=self.device) * self.config.v_std
        w_noise = torch.randn(K, T, device=self.device) * self.config.w_std
        #this generate [1000 56] random noise at std dev of respective vals

        # Add to nominal (broadcasting: [T] -> [K, T])
        # ideally i have [T] values -> [1 , T] to tell torch this is single sequence for broadcast
        v_samples = nominal_sequence.vx.unsqueeze(0) + v_noise
        w_samples = nominal_sequence.wz.unsqueeze(0) + w_noise

        # Clamp to limits
        v_samples = torch.clamp(v_samples, self.config.v_min, self.config.v_max)
        w_samples = torch.clamp(w_samples, -self.config.w_max, self.config.w_max)

        return v_samples, w_samples
    
    


