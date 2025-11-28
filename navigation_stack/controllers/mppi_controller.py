# In: navigation_stack/controllers/mppi_controller.py

import torch
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .mppi_types import MPPIConfig, MPPIState, ControlSequence
from .mppi_noise import NoiseGenerator
from .mppi_rollout import TrajectoryRollout
from .mppi_critic_manager import CriticManager

    

class MPPIController:
    """
    MPPI controller with integrated path management

    pass the full global path once during initialization.
    """

    def __init__(
        self,
        config: MPPIConfig,
        full_path: np.ndarray,    # full path
        goal: np.ndarray,         # [2] - final goal position
        ):
        """
        Initialize MPPI controller with full path.

        Args: 
            config: MPPI configuration
            full_path [N, 2] Full global path from start to goal
            goal: [2] Final goal position (x, y)
        """
        self.config = config
        self.device = config.device

        self.full_path = full_path   # [N, 2] numpy array
        self.goal = goal             # [2] numpy array


        # path tracking state
        self.last_chunk_update_pos = None
        self.current_path_chunk = None # update dynamically

        self.control_sequence = ControlSequence(self.config.horizon_steps, self.device)
        self.control_sequence.initalize_with_forward_motion(v=0.2)
        self.noise_generator = NoiseGenerator(self.config)
        self.trajectory_rollout = TrajectoryRollout(self.config)
        self.critic_manager = CriticManager(
            v_max=self.config.v_max,
            v_min=self.config.v_min,
            w_max=self.config.w_max,
            resolution=0.1, 
        )

    def _find_closest_waypoint_index(
        self,
        robot_position: np.ndarray
    )-> int:
        """
        Find the index of the closest waypoint on the path.

        Args:
            robot_position: [x, y] current position
        Returns:
            Index of closest waypoint
        """
        if len(self.full_path) == 0:
            return 0
        
        distances = np.linalg.norm(self.full_path - robot_position[:2], axis=1)
        closest_idx = np.argmin(distances)

        return int(closest_idx)
    
    def _extract_path_chunk(
        self,
        robot_position: np.ndarray
    )-> np.ndarray:
        """
        Extract relevant path chunk ahead of robot.

        Args:
            robot_position: [x, y] current position
        Returns:
            Path chunk [M, 2] where M>= min_path_points
        """

        # Find closest point
        closest_idx = self._find_closest_waypoint_index(robot_position)

        # Extract chunk from closest point to end 
        chunk = self.full_path[closest_idx:]

        #If chunk is too small, pad with final goal
        if len(chunk) < self.config.min_path_points:
            goal_2d = self.goal.reshape(1,2)
            padding_needed = self.config.min_path_points - len(chunk)
            padding = np.repeat(goal_2d, padding_needed, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)

        # Limit to Lookahead distance
        chunk = chunk[:self.config.lookahead_points]

        return chunk
    
    def _should_update_path_chunk(
        self,
        robot_position: np.ndarray
    )-> bool:
        """
        Check if we should recompute the path chunk.
        
        Update when:
        1. First call (no previous position)
        2. Robot moved significantly
        
        Args:
            robot_position: [x, y] current position
        
        Returns:
            True if should update chunk
        """
        if self.last_chunk_update_pos is None:
            return True
        
        distance_moved = np.linalg.norm(
            robot_position[:2] - self.last_chunk_update_pos[:2] 
        )

        return distance_moved>= self.config.path_update_distance
    
    def compute_control_command(
        self,
        current_pose: torch.Tensor,     # [3] - X Y Theta
        costmap: torch.Tensor,          # [H, W] - STVL costmap
        grid_origin: torch.Tensor       # [2] - costmap origin
    )-> Tuple[float, float]:
        """
        Compute MPPI control command.
        
        Get relevent path chunk based on robot position 
        Args: 
            current_pose [x, y, theta] current robot pose
            costmap: [H, W] STVL costmap (values 0-1)
            grid_origin: [2] costmap origin in world frame
        Returns:
            (v, w)- linear and angular velocity commands 
        """
        # Convert current pose to numpy for path processing

        # print("🐛 DEBUG: Entering compute_control_command")
        # print(f"   current_pose type: {type(current_pose)}")
        # print(f"   costmap type: {type(costmap)}")
        current_pos_np = current_pose[:2].cpu().numpy()
        distance_to_goal = np.linalg.norm(current_pos_np - self.goal)

        GOAL_TOLERANCE = 0.3
        if distance_to_goal < GOAL_TOLERANCE:
            return 0.0, 0.0  # ← STOP!
        # Extract path chunk if needed

        if self._should_update_path_chunk(current_pos_np):
            self.current_path_chunk = self._extract_path_chunk(current_pos_np)
            self.last_chunk_update_pos = current_pos_np.copy()

        # Convert path chunk to tensor

        path_chunk_tensor = torch.tensor(
            self.current_path_chunk,
            dtype=torch.float32,
            device=self.device
        )

        # print(f"🔍 Path chunk debug:")
        # print(f"   current_path_chunk.shape: {self.current_path_chunk.shape}")
        # print(f"   path_chunk_tensor.shape: {path_chunk_tensor.shape}")
        # # print(f"   path_chunk_tensor.ndim: {path_chunk_tensor.ndim}")
        # if len(path_chunk_tensor) > 0:
        #     print(f"   path_chunk_tensor[0]: {path_chunk_tensor[0]}")
        # if len(path_chunk_tensor) > 4:
        #     print(f"   path_chunk_tensor[4]: {path_chunk_tensor[4]}")
        goal_tensor = torch.tensor(
            self.goal,
            dtype=torch.float32,
            device=self.device
        )

        # print("🐛 DEBUG: Generating noise...")
        # Generate the v/w samples [K, T] ~ [1000, 56 ideally]
        v_samples, w_samples = self.noise_generator.generate_noisy_controls(self.control_sequence)
        # print("🐛 DEBUG: Rolling out trajectories...")
        # Trajectory rollout with xy theta vals 
        trajectories = self.trajectory_rollout.rollout(current_pose, v_samples, w_samples)
        # print(f"   trajectories type: {type(trajectories)}")
        # print(f"   trajectories shape: {trajectories.shape}")

        # print("🐛 DEBUG: Evaluating costs...")
        total_costs = self.critic_manager.evaluate_trajectories(trajectories,
                                          v_samples,
                                          w_samples,
                                          path_chunk_tensor,
                                          goal_tensor,
                                          current_pose,
                                          costmap,
                                          grid_origin,
                                          self.config.dt)
        min_cost = torch.min(total_costs)
        weights = torch.exp(-(total_costs-min_cost) / self.config.lambda_)
        weights = weights / torch.sum(weights) # Normalise
        
        # Weighted Average along Control Samples
        # High note we have 1000 56 values of we have weights which are sum along over respected horizon
        # Now those [K] taken broadcasted at [K, 1] ~ converted to cols then whe we multiply with v sampled 
        # we have weighted velocity along all trajectory and horizon now we take sum along all trajectory at repected t
        v_new = torch.einsum('k,kt->t', weights, v_samples)  # [T]
        w_new = torch.einsum('k,kt->t', weights, w_samples)  # [T]

        # Update control sequence
        self.control_sequence.vx = v_new
        self.control_sequence.wz = w_new
        self.control_sequence.clamp(self.config.v_min, self.config.v_max, self.config.w_max)

        v, w = self.control_sequence.get_first_command()
        self.control_sequence.shift()   # warm_start for next iteration

        return v, w
    
    def get_progress(self) -> dict:
        """
        Get current progress along the path.
        
        Returns:
            Dictionary with progress information
        """
        if self.last_chunk_update_pos is None:
            return {
                "progress_pct": 0.0,
                "distance_to_goal": float('inf'),
                "waypoints_remaining": len(self.full_path)
            }
        
        closest_idx = self._find_closest_waypoint_index(self.last_chunk_update_pos)
        progress_pct = (closest_idx / len(self.full_path)) * 100
        
        distance_to_goal = np.linalg.norm(
            self.last_chunk_update_pos[:2] - self.goal
        )
        
        return {
            "progress_pct": progress_pct,
            "distance_to_goal": distance_to_goal,
            "waypoints_remaining": len(self.full_path) - closest_idx,
            "current_waypoint_idx": closest_idx
        }



