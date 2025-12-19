# In: navigation_stack/controllers/mppi_rollout.py

import torch
from .mppi_types import MPPIConfig

class TrajectoryRollout:
    """
    Forward-simulates the differntial drive robot model.
    """
    def __init__(self, config: MPPIConfig):
        self.config = config
        self.dt = config.dt

    def rollout(
            self,
            initial_state: torch.Tensor,
            v_samples: torch.Tensor, #[K, T]
            w_samples: torch.Tensor  #[K, T]
    ) -> torch.Tensor:
        """
        Integrates K Trajectories in parallel.
        Returns: trajectories [K, T ,3] where last dim is (x, y, theta)

        """
        K = v_samples.shape[0] # num_sample
        T = v_samples.shape[1] # horizon_steps
        device = self.config.device

        # Initialize Storage: [K, T, 3]
        trajectories = torch.zeros(K, T , 3, device=device)

        # All K samples start from the same state
        # initial_state = torch.tensor([robot_X, robot_Y, robot_theta])
        x = torch.full((K,), initial_state[0].item(), device=device)
        y = torch.full((K,), initial_state[1].item(), device=device)
        theta = torch.full((K,), initial_state[2].item(), device=device)

        # Integrate Forward (only horizon steps)
        for t in range(T):
            # Get controls at this timestep: [K]
            v =v_samples[:, t]
            w =w_samples[:, t] # At all 1000 trajectory pick given index v / w at time step t


            x = x + v * torch.cos(theta) * self.dt
            y = y + v * torch.sin(theta) * self.dt

            # Differntial drive kinematics (vectorized over K!)
            theta = theta + w * self.dt

            # Normalise angle to [-pi to pi]
            

            theta = torch.atan2(torch.sin(theta), torch.cos(theta))

            # Store this timestep: [K, 3]
            trajectories[:, t, 0] = x
            trajectories[:, t ,1] = y
            trajectories[:, t, 2] = theta

        return trajectories

