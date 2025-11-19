# In: navigation_stack/controllers/mppi_controller.py

import sys
import os
import torch
import numpy as np
import yaml  # <-- NEW IMPORT
from pathlib import Path


# # --- Add project root to path ---
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CONTROLLERS_ROOT = SCRIPT_DIR
# STACK_ROOT = os.path.dirname(CONTROLLERS_ROOT)
# PROJECT_ROOT = os.path.dirname(STACK_ROOT)
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
# ---------------------------------

# --- Auto-select GPU if available ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("MPPI Controller: Using GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("MPPI Controller: Warning! CUDA not available, using CPU.")
# ------------------------------------


class MPPIController:
    """
    A PyTorch-based MPPI (Model Predictive Path Integral) controller
    for a differential drive robot.
    """
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to the config file in the package
            config_path = Path(__file__).parent.parent / "configs" / "mppi_config.yaml"
        print("--- Initializing PyTorch MPPI Controller ---")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Simulation parameters
        self.K = config['simulation']['num_samples']
        self.T = config['simulation']['horizon_steps']
        self.dt = config['simulation']['dt']
        self.lambda_ = config['simulation']['lambda_']
        
        # Robot limits
        self.v_max = config['robot_limits']['max_linear_vel']
        self.w_max = config['robot_limits']['max_angular_vel']
        self.safety_radius = config['robot_limits']['safety_radius']
        
        # Noise parameters
        self.v_std = config['noise_params']['v_std']
        self.w_std = config['noise_params']['omega_std']
        
        # Cost weights
        self.w_track = config['cost_weights']['tracking']
        self.w_obs = config['cost_weights']['obstacle']
        self.w_goal = config['cost_weights']['goal']
        self.w_control = config['cost_weights']['control_cost']
        self.w_smooth = config['cost_weights']['smoothing']
        
        # Grid parameters
        self.grid_resolution = config['grid_params']['resolution']
        self.grid_width_m = config['grid_params']['width_m']
        self.grid_height_m = config['grid_params']['height_m']

        # Calculate grid dimensions
        self.grid_width_cells = int(self.grid_width_m / self.grid_resolution)
        self.grid_height_cells = int(self.grid_height_m / self.grid_resolution)
        
        # Grid origin (bottom-left corner in world coordinates)
        self.grid_origin = torch.tensor([
            -(self.grid_width_cells / 2.0) * self.grid_resolution,
            -(self.grid_height_cells / 2.0) * self.grid_resolution
        ], device=DEVICE, dtype=torch.float32)
        
        # Previous control (for smoothing)
        self.prev_v = 0.15
        self.prev_w = 0.0
        
        # Control sequence for warm-starting
        self.prev_v_sequence = torch.zeros(self.T, device=DEVICE)
        self.prev_w_sequence = torch.zeros(self.T, device=DEVICE)
        
        # Stats
        self.nan_count = 0
        self.iteration = 0
        
        print(f"--- MPPIController Initialized ---")
        print(f"  Samples: {self.K}, Horizon: {self.T}, dt: {self.dt}")
        print(f"  Grid: {self.grid_width_cells}x{self.grid_height_cells} @ {self.grid_resolution}m")
        print(f"  Grid Origin: {self.grid_origin.cpu().numpy()}")
        print(f"  Weights: track={self.w_track}, obs={self.w_obs}, goal={self.w_goal}")
        print("----------------------------------")

    # def _load_config(self, config_name):
    #     """Loads controller parameters from our YAML file."""
    #     try:
    #         with ExitStack() as stack:
    #             # Find the 'navigation_stack.configs' package
    #             configs_pkg = resources.files("navigation_stack.configs")
    #             # Open the resource file
    #             yaml_file = stack.enter_context(resources.as_file(configs_pkg.joinpath(config_name)))

    #             with open(yaml_file, 'r') as f:
    #                 config = yaml.safe_load(f)
            
    #         print(f"Loaded config: {config_name}")

    #         grid = config['grid_params']
    #         self.grid_resolution = grid['resolution']
    #         self.grid_width_cells = int(grid['width_m'] / self.grid_resolution)
    #         self.grid_height_cells = int(grid['height_m'] / self.grid_resolution)

    #         # This is the (x, y) coordinate of the bottom-left corner
    #         self.grid_origin = torch.tensor(
    #             [- (self.grid_width_cells / 2.0) * self.grid_resolution, 
    #             - (self.grid_height_cells / 2.0) * self.grid_resolution],
    #             device=DEVICE, dtype=torch.float32
    #         )

    #         # --- Store all parameters from config ---
    #         sim_params = config['simulation']
    #         self.K = sim_params['num_samples']
    #         self.T = sim_params['horizon_steps']
    #         self.dt = sim_params['dt']
    #         self.lambda_ = sim_params['lambda_']
            
    #         limits = config['robot_limits']
    #         self.v_max = limits['max_linear_vel']
    #         self.omega_max = limits['max_angular_vel']
    #         self.safety_radius = limits['safety_radius']
            
    #         noise = config['noise_params']
    #         self.v_std = noise['v_std']
    #         self.omega_std = noise['omega_std']
            
    #         weights = config['cost_weights']
    #         self.w_track = weights['tracking']
    #         self.w_obs = weights['obstacle']
    #         self.w_goal = weights['goal']
    #         self.w_control = weights['control_cost']
    #         self.w_smooth = weights['smoothing']
            
    #     except Exception as e:
    #         print(f"FATAL ERROR: Could not load config file '{config_name}'")
    #         print(f"Make sure it is in 'navigation_stack/configs/'")
    #         print(f"Error: {e}")
    #         sys.exit(1)

    # def _normalize_angles(self, angles):
    #     """Normalize angles to be within [-pi, pi]."""
    #     return (angles + torch.pi) % (2 * torch.pi) - torch.pi
    
    # def _world_to_grid(self, points_xy):
    #     """Converts (..., 2) world (x, y) points to (..., 2) grid (col, row) indices."""
    #     # Shift points by origin and scale by resolution
    #     indices = (points_xy - self.grid_origin) / self.grid_resolution
    #     return torch.floor(indices).long()
    

    # def _generate_control_sequences(self):
    #     """
    #     Generates K noisy control sequences based on the previous optimal sequence.
        
    #     Returns:
    #         torch.Tensor: Shape [K, T, 2]
    #     """
        
    #     # Return a tensor of zeros with the correct shape
    #     # 1. Create a tensor of noise
    #     # Shape: [K, T, 2] -> [v_noise, omega_noise] for all samples, all timesteps
    #     noise = torch.randn(self.K, self.T, 2, device=DEVICE)
        
    #     # 2. Scale the noise by our standard deviations
    #     # We create a [1, 1, 2] tensor to broadcast correctly
    #     std_devs = torch.tensor([self.v_std, self.omega_std], device=DEVICE).view(1, 1, 2)
    #     noise = noise * std_devs
        
    #     # 3. Add noise to the previous (best) control sequence
    #     # self.prev_control_sequence has shape [T, 2]
    #     # PyTorch broadcasting automatically adds it to all K samples
    #     # Result: [K, T, 2]
    #     noisy_sequences = self.prev_control_sequence + noise
        
    #     # 4. Clamp (limit) the controls to the robot's physical capabilities
        
    #     # Clamp linear velocity (v): [0, v_max]
    #     # [..., 0] selects all 'v' values
    #     noisy_sequences[..., 0] = torch.clamp(
    #         noisy_sequences[..., 0], 0.0, self.v_max
    #     )
        
    #     # Clamp angular velocity (omega): [-omega_max, omega_max]
    #     # [..., 1] selects all 'omega' values
    #     noisy_sequences[..., 1] = torch.clamp(
    #         noisy_sequences[..., 1], -self.omega_max, self.omega_max
    #     )
        
    #     return noisy_sequences

    def compute_control_command(self, current_pose, path, height_map):
        """
        Clean MPPI implementation - no aggressive smoothing or biasing
        Let the cost function do all the work
        """
        self.iteration += 1
        
        if len(path) == 0:
            return [0.0, 0.0]
        
        try:
            # Convert to tensors
            pose = torch.tensor(current_pose, dtype=torch.float32, device=DEVICE)
            path_array = np.array(path, dtype=np.float32)
            path_tensor = torch.tensor(path_array, dtype=torch.float32, device=DEVICE)
            
            if path_tensor.dim() == 1:
                path_tensor = path_tensor.unsqueeze(0)
            
            # Simple warm-started sampling (like the proven implementation)
            v_samples = (self.prev_v_sequence.unsqueeze(0).expand(self.K, -1).clone() + 
                        torch.randn(self.K, self.T, device=DEVICE) * self.v_std)
            w_samples = (self.prev_w_sequence.unsqueeze(0).expand(self.K, -1).clone() + 
                        torch.randn(self.K, self.T, device=DEVICE) * self.w_std)
            
            # Clamp to limits
            v_samples = torch.clamp(v_samples, 0.0, self.v_max)
            w_samples = torch.clamp(w_samples, -self.w_max, self.w_max)
            
            # Rollout and compute costs
            costs = self._compute_trajectory_costs(pose, v_samples, w_samples, 
                                                path_tensor, height_map)
            
            # Check for NaN
            if torch.any(torch.isnan(costs)) or torch.any(torch.isinf(costs)):
                print(f"WARNING: NaN/Inf in costs")
                return self._fallback_control(current_pose, path[0])
            
            # Normalize costs for numerical stability
            cost_min = torch.min(costs)
            cost_max = torch.max(costs)
            
            if cost_max - cost_min < 1e-6:
                # All costs nearly identical - use best sample
                best_idx = torch.argmin(costs)
                v_opt = v_samples[best_idx, 0].item()
                w_opt = w_samples[best_idx, 0].item()
            else:
                # Standard MPPI weighting
                costs_normalized = (costs - cost_min) / (cost_max - cost_min + 1e-8)
                weights = torch.exp(-costs_normalized / (self.lambda_ + 1e-8))
                weights = weights / (torch.sum(weights) + 1e-8)
                
                # Weighted average of first timestep controls
                v_opt = torch.sum(weights * v_samples[:, 0]).item()
                w_opt = torch.sum(weights * w_samples[:, 0]).item()
            
            # Check output
            if np.isnan(v_opt) or np.isnan(w_opt):
                return self._fallback_control(current_pose, path[0])
            
            # Update warm-start sequences (shift and append)
            best_idx = torch.argmin(costs)
            self.prev_v_sequence = v_samples[best_idx]
            self.prev_w_sequence = w_samples[best_idx]
            
            # NO extra smoothing - return raw MPPI output
            v_cmd = v_opt
            w_cmd = w_opt
            
            # Final sanity check
            if np.isnan(v_cmd) or np.isnan(w_cmd) or np.isinf(v_cmd) or np.isinf(w_cmd):
                return self._fallback_control(current_pose, path[0])
            
            self.prev_v = v_cmd
            self.prev_w = w_cmd
            self.nan_count = 0
            
            # Debug output
            if self.iteration % 100 == 0:
                target = path_tensor[0]
                dx = target[0] - pose[0]
                dy = target[1] - pose[1]
                dist_to_target = torch.sqrt(dx**2 + dy**2).item()
                print(f"  MPPI: dist={dist_to_target:.2f}m, cost_range=[{cost_min:.1f}, {cost_max:.1f}]")
        
            return [v_cmd, w_cmd]
        
        except Exception as e:
            print(f"ERROR in MPPI: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_control(current_pose, path[0] if len(path) > 0 else (0, 0))

        
    def _fallback_control(self, current_pose, target):
        """Simple proportional controller fallback"""
        self.nan_count += 1
        
        if self.nan_count > 10:
            print(f"CRITICAL: Fallback used {self.nan_count} times!")
            return [0.0, 0.0]  # Stop the robot
        
        print(f"Using fallback control (#{self.nan_count})")
        
        dx = target[0] - current_pose[0]
        dy = target[1] - current_pose[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = angle_to_target - current_pose[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        if abs(angle_diff) > 0.3:
            v = 0.05
            w = np.clip(1.5 * angle_diff, -self.w_max, self.w_max)
        else:
            v = np.clip(0.3 * distance, 0.0, self.v_max)
            w = np.clip(0.8 * angle_diff, -self.w_max, self.w_max)
        
        self.prev_v = v
        self.prev_w = w
        
        return [v, w]

    
    # def _simulate_rollouts(self, current_state, control_sequences):
    #     """
    #     Simulates all K trajectories forward in time.
    #     This is the physics model, run in parallel on the GPU.
        
    #     Args:
    #         current_state (torch.Tensor): Shape [3] -> [x, y, theta]
    #         control_sequences (torch.Tensor): Shape [K, T, 2]

    #     Returns:
    #         torch.Tensor: Shape [K, T, 3] (all simulated paths)
    #     """
    #     # We need to store all states [x, y, theta] for all K samples 
    #     # over all T timesteps.
    #     # Shape: [K, T, 3]
    #     sampled_trajectories = torch.zeros(self.K, self.T, 3, device=DEVICE)
        
    #     # We need to broadcast the single [3] current_state to be the
    #     # starting point for all K samples.
    #     # current_state_k has shape [K, 3]
    #     current_state_k = current_state.unsqueeze(0).repeat(self.K, 1)
        
    #     # Loop over the time horizon T (this is fast)
    #     for t in range(self.T):
    #         # Get the controls for this timestep
    #         # controls has shape [K, 2]
    #         controls = control_sequences[:, t, :]
            
    #         # Extract v and omega (each shape [K])
    #         v = controls[:, 0]
    #         omega = controls[:, 1]
            
    #         # Extract previous state (shape [K])
    #         prev_x = current_state_k[:, 0]
    #         prev_y = current_state_k[:, 1]
    #         prev_theta = current_state_k[:, 2]
            
    #         # --- Apply Differential Drive Kinematics (Vectorized) ---
            
    #         # 1. Calculate new theta
    #         new_theta = self._normalize_angles(prev_theta + omega * self.dt)
            
    #         # 2. Calculate new x and y
    #         new_x = prev_x + v * torch.cos(new_theta) * self.dt
    #         new_y = prev_y + v * torch.sin(new_theta) * self.dt
            
    #         # 3. Stack them back into a [K, 3] tensor
    #         new_states = torch.stack([new_x, new_y, new_theta], dim=1)
            
    #         # 4. Save this step's result
    #         sampled_trajectories[:, t, :] = new_states
            
    #         # 5. The new state becomes the current state for the next loop
    #         current_state_k = new_states
            
    #     return sampled_trajectories
    
    # def _compute_obstacle_cost(self, robot_positions, height_map):
    #     """Calculates the cost from the 2.5D Height Map."""

    #     # 1. Convert all simulated world (x, y) coords to grid (col, row) indices
    #     # grid_indices will be shape [K, T, 2]
    #     grid_indices = self._world_to_grid(robot_positions)

    #     # 2. Filter out-of-bounds indices
    #     col_indices = grid_indices[..., 0]
    #     row_indices = grid_indices[..., 1]

    #     # Create a mask for valid indices
    #     mask = (col_indices >= 0) & (col_indices < self.grid_width_cells) & \
    #         (row_indices >= 0) & (row_indices < self.grid_height_cells)

    #     # 3. Look up costs
    #     # Start with a high cost for all points
    #     # (This penalizes trajectories that go "out of bounds")
    #     cost_per_timestep = torch.full_like(col_indices, 100.0, dtype=torch.float32, device=DEVICE)

    #     # Get the indices that are valid
    #     valid_cols = col_indices[mask]
    #     valid_rows = row_indices[mask]

    #     # height_map uses (row, col)
    #     # Look up the height (cost) from the map for all valid points
    #     looked_up_costs = height_map[valid_rows, valid_cols]

    #     # "Scatter" the real costs back into the tensor
    #     cost_per_timestep[mask] = looked_up_costs

    #     # 4. Final cost
    #     # We square the cost to *heavily* penalize taller obstacles
    #     # (e.g., a 2m wall is much worse than a 0.1m curb)
    #     total_cost = self.w_obs * cost_per_timestep

    #     # Sum cost over time for each trajectory
    #     return torch.sum(total_cost, dim=1)
    
    # def _compute_tracking_cost(self, robot_positions):
    #     """
    #     Computes the Cross-Track Error cost.
    #     Penalizes trajectories for being far from the reference path.
    #     """
    #     # Get the reference path nodes from the class
    #     path_nodes = self.path_tensor  # Shape: (N, 2)

    #     # Calculate the distance from EVERY trajectory point to EVERY path node
    #     # pos_expanded shape: (K, T, 1, 2)
    #     pos_expanded = robot_positions.unsqueeze(2)
    #     # path_expanded shape: (1, 1, N, 2)
    #     path_expanded = path_nodes.unsqueeze(0).unsqueeze(0)

    #     # dists is shape (K, T, N)
    #     dists = torch.norm(pos_expanded - path_expanded, dim=3)

    #     # Find the minimum distance (distance to closest path node)
    #     # min_dists is shape (K, T)
    #     min_dists, _ = torch.min(dists, dim=2)

    #     # Average this "off-path" cost over the horizon
    #     cost = torch.mean(min_dists, dim=1) # Shape: (K,)
    #     return self.w_track * cost

    # def _compute_goal_cost(self, robot_positions):
    #     """
    #     Computes the cost of not reaching the FINAL goal.
    #     This "pulls" the robot toward the end of the entire path.
    #     """
    #     # Get all final [x, y] positions: [K, 2]
    #     final_positions = robot_positions[:, -1, :]

    #     # Get the FINAL goal from the path list
    #     final_goal = self.path_tensor[-1] # Shape: (2,)

    #     # Get distance from final point to target: [K]
    #     dist_to_goal = torch.norm(final_positions - final_goal, dim=1)

    #     return self.w_goal * dist_to_goal

    # def _compute_control_cost(self, control_sequences):
    #     """Calculates the cost for high-effort controls."""
    #     # Sum of squares of v and omega over time: [K]
    #     return self.w_control * torch.sum(control_sequences**2, dim=[1, 2])

    # def _compute_smoothing_cost(self, control_sequences):
    #     """Calculates the cost for jerky movements."""
    #     # Calculate differences between controls at t and t-1: [K, T-1, 2]
    #     control_diffs = torch.diff(control_sequences, n=1, dim=1)
        
    #     # Sum of squares of the changes: [K]
    #     return self.w_smooth * torch.sum(control_diffs**2, dim=[1, 2])
    

    
    # def _compute_all_costs(self, sampled_trajectories, control_sequences, height_map):
    #     """
    #     Calculates the cost for every timestep in every trajectory.
    #     This is the "cost function" logic, run in parallel on the GPU.

    #     Args:
    #         sampled_trajectories (torch.Tensor): Shape [K, T, 3]
    #         control_sequences (torch.Tensor): Shape [K, T, 2]
    #         height_map (torch.Tensor): Shape [H, W] (The 2.5D grid)

    #     Returns:
    #         torch.Tensor: Shape [K] (The total cost for each trajectory)
    #     """
    #     # Get all [x, y] positions from trajectories [K, T, 2]
    #     robot_positions = sampled_trajectories[..., :2]

    #     # Calculate each component cost
    #     obs_cost = self._compute_obstacle_cost(robot_positions, height_map) 
    #     track_cost = self._compute_tracking_cost(robot_positions)
    #     goal_cost = self._compute_goal_cost(robot_positions)
    #     control_cost = self._compute_control_cost(control_sequences)
    #     smooth_cost = self._compute_smoothing_cost(control_sequences)

    #     # Sum all costs
    #     total_costs = (
    #         obs_cost + 
    #         track_cost + 
    #         goal_cost + 
    #         control_cost + 
    #         smooth_cost
    #     )

    #     return total_costs



    
    # def compute_control_command(self, current_pose, path_list, height_map):
    #     """
    #     The main public function that computes the optimal control.

    #     Args:
    #         current_pose (np.array): [x, y, theta]
    #         path_list (list): The list of [x, y] waypoints to follow
    #         height_map (torch.Tensor): The [H, W] 2.5D cost grid

    #     Returns:
    #         np.array: [v, omega] the optimal command
    #     """

    #     # --- 0. Convert all inputs to PyTorch Tensors on the GPU ---
    #     state = torch.tensor(current_pose, dtype=torch.float32, device=DEVICE)
    #     self.path_tensor = torch.tensor(path_list, dtype=torch.float32, device=DEVICE)

    #     # The height_map is already a tensor, so we just use it directly
    #     obstacles = height_map


    #     # --- 1. Generate K noisy control sequences (the "what ifs") ---
    #     control_sequences = self._generate_control_sequences()

    #     # --- 2. Simulate all K trajectories (the "rollouts") ---
    #     sampled_trajectories = self._simulate_rollouts(state, control_sequences)

    #     # --- 3. Compute cost for all K trajectories ---
    #     costs = self._compute_all_costs(sampled_trajectories, control_sequences, obstacles)       

    #     # ... (rest of the function is identical) ...
    #     min_cost = torch.min(costs)
    #     weights = torch.exp(-self.lambda_ * (costs - min_cost))
    #     weights = weights / (torch.sum(weights) + 1e-10) # Normalize

    #     # Reshape weights for broadcasting: [K, 1, 1]
    #     weights = weights.view(self.K, 1, 1)

    #     # Calculate weighted average of all control sequences
    #     optimal_sequence = torch.sum(weights * control_sequences, dim=0)

    #     # 5. Save the new optimal sequence for warm start
    #     shifted_sequence = torch.roll(optimal_sequence, shifts=-1, dims=0)
    #     shifted_sequence[-1] = optimal_sequence[-1] # Repeat last action
    #     self.prev_control_sequence = shifted_sequence

    #     # 6. Return command as a numpy array
    #     best_command_tensor = optimal_sequence[0, :]

    #     return best_command_tensor.cpu().numpy()

    def _compute_trajectory_costs(self, pose, v_samples, w_samples, path, height_map):
        """
        Fully parallel trajectory rollout - NO loops!
        Computes all K samples × T timesteps simultaneously
        """
        K = v_samples.shape[0]
        T = v_samples.shape[1]
        
        # Initialize states: (K, T+1, 3) - we need T+1 to store initial state
        # Shape: [K samples, T+1 timesteps, 3 state dimensions (x, y, theta)]
        states = torch.zeros(K, T + 1, 3, device=DEVICE)
        states[:, 0, :] = pose.unsqueeze(0).expand(K, -1)  # All start from current pose
        
        # Vectorized rollout for ALL timesteps at once
        for t in range(T):
            # Extract current state
            theta = states[:, t, 2]  # (K,)
            
            # Get controls for this timestep
            v = v_samples[:, t]  # (K,)
            w = w_samples[:, t]  # (K,)
            
            # Update ALL K samples for this timestep in parallel
            states[:, t+1, 0] = states[:, t, 0] + v * torch.cos(theta) * self.dt
            states[:, t+1, 1] = states[:, t, 1] + v * torch.sin(theta) * self.dt
            states[:, t+1, 2] = torch.atan2(torch.sin(states[:, t, 2] + w * self.dt), 
                                            torch.cos(states[:, t, 2] + w * self.dt))
        
        # Now compute costs on ALL states at once (K, T, 3)
        # Remove initial state, keep only the T predicted states
        predicted_states = states[:, 1:, :]  # (K, T, 3)
        
        # Compute ALL costs in parallel
        positions = predicted_states[:, :, :2]  # (K, T, 2)
        
        # Tracking cost: distance to path for ALL K×T positions at once
        # Reshape to (K*T, 2) for cdist, then reshape back
        positions_flat = positions.reshape(-1, 2)  # (K*T, 2)
        distances = torch.cdist(positions_flat, path)  # (K*T, P)
        min_distances = torch.min(distances, dim=1)[0]  # (K*T,)
        min_distances = min_distances.reshape(K, T)  # (K, T)
        tracking_costs = self.w_track * torch.sum(min_distances**2, dim=1)  # (K,)
        
        # Obstacle cost: lookup ALL K×T positions at once
        grid_coords = ((positions_flat - self.grid_origin) / self.grid_resolution).long()
        grid_coords[:, 0] = torch.clamp(grid_coords[:, 0], 0, self.grid_width_cells - 1)
        grid_coords[:, 1] = torch.clamp(grid_coords[:, 1], 0, self.grid_height_cells - 1)
        heights = height_map[grid_coords[:, 1], grid_coords[:, 0]]  # (K*T,)
        heights = heights.reshape(K, T)  # (K, T)
        obstacle_costs = self.w_obs * torch.sum(
            torch.where(heights > 0.15, 1000.0, 0.0), dim=1
        )  # (K,)
        
        # Goal cost: final positions only
        final_positions = predicted_states[:, -1, :2]  # (K, 2)
        goal_distances = torch.norm(final_positions - path[-1], dim=1)  # (K,)
        goal_costs = self.w_goal * goal_distances
        
        # Control cost: energy over entire trajectory
        control_costs = self.w_control * torch.sum(v_samples**2 + w_samples**2, dim=1)  # (K,)
        
        # Smoothing cost: changes between timesteps
        v_diffs = torch.diff(v_samples, dim=1)  # (K, T-1)
        w_diffs = torch.diff(w_samples, dim=1)  # (K, T-1)
        smoothing_costs = self.w_smooth * torch.sum(v_diffs**2 + w_diffs**2, dim=1)  # (K,)
        
        # Total cost
        total_costs = tracking_costs + obstacle_costs + goal_costs + control_costs + smoothing_costs
        
        return total_costs
    
    def _tracking_cost(self, states, path):
        """Distance to closest path point"""
        positions = states[:, :2]  # (K, 2)
        distances = torch.cdist(positions, path)  # (K, P)
        min_distances = torch.min(distances, dim=1)[0]
        return self.w_track * min_distances**2
    

    def _obstacle_cost(self, states, height_map):
        """Cost from height map obstacles"""
        K = states.shape[0]
        positions = states[:, :2]
        
        # Convert to grid coordinates
        grid_coords = ((positions - self.grid_origin) / self.grid_resolution).long()
        
        # Clamp to bounds
        grid_coords[:, 0] = torch.clamp(grid_coords[:, 0], 0, self.grid_width_cells - 1)
        grid_coords[:, 1] = torch.clamp(grid_coords[:, 1], 0, self.grid_height_cells - 1)
        
        # Lookup heights
        heights = height_map[grid_coords[:, 1], grid_coords[:, 0]]
        
        # High cost for obstacles
        obstacle_cost = torch.where(
            heights > 0.15,  # Obstacle threshold
            torch.full_like(heights, 1000.0),  # High penalty
            torch.zeros_like(heights)
        )
        
        return self.w_obs * obstacle_cost

    def _goal_cost(self, states, goal):
        """Rewards getting closer to the goal."""  # <-- Says REWARD but...
        positions = states[:, :2]
        distances = torch.norm(positions - goal, dim=1)
        return self.w_goal * distances  # <-- This PENALIZES being far!





# --- Test Script ---
# if __name__ == "__main__":
#     """
#     This block allows us to test the controller directly
#     by running: python navigation_stack/controllers/mppi_controller.py
#     """
#     print("\n--- Testing MPPI Controller Full Computation ---")
    
#     controller = MPPIController(config_name="mppi_config.yaml")
    
#     # Test 1: Obstacle in the way
#     pose = np.array([0.0, 0.0, 0.0])
#     waypoint = np.array([[2.0, 0.0]]) 
#     lidar = np.array([[1.0, 0.0]]) # Obstacle 1m ahead
    
#     print(f"\nTest 1: Target straight ahead, obstacle in the way.")
#     cmd = controller.compute_control_command(pose, waypoint, lidar)
#     print(f"Computed command (should NOT be [0, 0]): {cmd}")
#     print(f"  (Expected: v > 0, omega != 0 to steer around obstacle)")
    
#     # Test 2: No obstacles
#     lidar_empty = np.array([])
    
#     print(f"\nTest 2: Target straight ahead, NO obstacles.")
#     cmd = controller.compute_control_command(pose, waypoint, lidar_empty)
#     print(f"Computed command (should NOT be [0, 0]): {cmd}")
#     print(f"  (Expected: v > 0, omega approx 0)")