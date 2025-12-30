# In: navigation_stack/controllers/mppi_controller.py

import torch
from typing import Tuple
import numpy as np
from .mppi_types import MPPIConfig, ControlSequence
from .mppi_noise import NoiseGenerator
from .mppi_rollout import TrajectoryRollout
from .mppi_critic_manager import CriticManager
from .path_tracker import PathTracker
    

class MPPIController:
    """
    MPPI controller with integrated path management and unified critic system.
    
    Matches Nav2 MPPI architecture:
    - PathTracker for monotonic progress tracking
    - CriticManager with unified CriticData structure
    - Efficient path validity computation (once per cycle)
    """

    def __init__(
        self,
        config: MPPIConfig,
        full_path: np.ndarray,    # full path #[N,3]
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
        self._debug_counter = 0

        # Initialize PathTracker (GPU-accelerated)
        self.path_tracker = PathTracker(
            full_path=full_path,
            goal=goal,
            device=self.device,
            verbose=False  # Set True for debugging
        )

        # Initialize control sequence
        self.control_sequence = ControlSequence(
            self.config.horizon_steps,
            self.device
        )
        self.control_sequence.initalize_with_forward_motion(v=0)

        # Initialize noise generator
        self.noise_generator = NoiseGenerator(self.config)

        # Initialize trajectory rollout
        self.trajectory_rollout = TrajectoryRollout(self.config)

        # Initialize CriticManager with unified interface
        self.critic_manager = CriticManager(
            # Obstacle avoidance
            cost_critic_weight=4.0,   # 3.81
            
            # Path following
            path_align_weight=0.4,    # 10.0
            path_angle_weight=0.9,     # 2.2
            path_follow_weight=2.0,     # 5.0
            
            # Goal reaching
            goal_weight=5.0,            # 5.0
            goal_angle_weight=3.0,      # 3.0
            
            # Motion quality
            constraint_weight=4,      # 2.0
            twirling_weight=4,        # 4.0
            prefer_forward_weight=2.0,     # 15.0

            speed_incentive_weight=0,  # ← NEGATIVE = REWARD!
            enable_speed_incentive=True,
            
            # Optional
            use_deadband_critic=False,


            
            # Parameters
            resolution=self.config.grid_resolution,
            
            # Enable/disable critics (for tuning)
            enable_cost_critic=True,
            enable_path_align=True,
            enable_path_angle=True,
            enable_path_follow=True,
            enable_goal=True,
            enable_goal_angle=True,
            enable_constraint=True,
            # enable_smoothness=True,
            enable_twirling=True,
            enable_prefer_forward=True,
            
            # Statistics
            publish_stats=False,  # Set True for debugging
            verbose=False
        )

        # In MPPIController.__init__():
        print("✅ MPPIController initialized successfully")
        print(f"   Device: {self.device}")
        print(f"   Path points: {len(full_path)}")
        print(f"   Horizon: {self.config.horizon_steps} steps")
    
    
    def _apply_smoothing(self, window_size=5):
        """
        Simple moving average smoothing (Simulating Savitzky-Golay)
        Applied to the control sequence to stop "snake" oscillation.
        """
        # We access the internal tensors of control_sequence
        vx = self.control_sequence.vx.cpu().numpy()
        wz = self.control_sequence.wz.cpu().numpy()
        
        # Apply simple convolution (moving average)
        # We only smooth if we have enough steps
        if len(vx) > window_size:
            kernel = np.ones(window_size) / window_size
            
            # Smooth VX
            vx_smooth = np.convolve(vx, kernel, mode='same')
            # Keep the ends valid (convolution messes up edges)
            vx_smooth[:2] = vx[:2] 
            vx_smooth[-2:] = vx[-2:]
            
            # Smooth WZ (Critical for snaking!)
            wz_smooth = np.convolve(wz, kernel, mode='same')
            wz_smooth[:2] = wz[:2]
            wz_smooth[-2:] = wz[-2:]
            
            # Update the tensors
            self.control_sequence.vx = torch.tensor(vx_smooth, device=self.device, dtype=torch.float32)
            self.control_sequence.wz = torch.tensor(wz_smooth, device=self.device, dtype=torch.float32)
    
    def compute_control_command(
        self,
        current_pose: torch.Tensor,     # [3] - X Y Theta
        costmap: torch.Tensor,          # [H, W] - STVL costmap
        grid_origin: torch.Tensor       # [2] - costmap origin
    )-> Tuple[float, float]:
        """
        Compute MPPI control command using unified critic system.
        
        This is the main control loop called at MPPI frequency (20 Hz).
        
        Args: 
            current_pose: [3] current robot pose (x, y, theta)
            costmap: [H, W] STVL costmap (values 0.0 to 1.0)
            grid_origin: [2] costmap origin in world frame (robot-centric!)
        
        Returns:
            (v, w): Linear and angular velocity commands
        """
        # Early Exit: check if goal reached

        # Check goal-> stop if reached early cycle call
        if self.path_tracker.is_near_goal(current_pose):
            print("GOAL REACHED!")
            return 0.0, 0.0


        # Update PathTracker (monotonic progress)
        self.path_tracker.update_progress(
            robot_position=current_pose[:2]
        )



        # Generate the v/w samples [K, T] ~ [1000, 56 ideally]
        v_samples, w_samples = self.noise_generator.generate_noisy_controls(
            self.control_sequence
        )
        
        # Rollout trajectories [K, T, 3]
        trajectories = self.trajectory_rollout.rollout(
            initial_state=current_pose,
            v_samples=v_samples,
            w_samples=w_samples
        )



        #Prepare unified CriticData
        # This is where path validity is computed (ONCE per cycle!)
        data = self.critic_manager.prepare_critic_data(
            trajectories=trajectories,
            v_samples=v_samples,
            w_samples=w_samples,
            path_tracker=self.path_tracker,
            costmap=costmap,
            grid_origin=grid_origin,
            current_pose=current_pose,
            dt=self.config.dt,
            v_max=self.config.v_max,
            v_min=self.config.v_min,
            w_max=self.config.w_max,
            compute_path_validity=True 
        )
        
        # Evaluate all critics with unified data

        total_costs = self.critic_manager.evaluate_trajectories(data)

        # Add Gamma regularization
        K, T = v_samples.shape
        gamma_vx = self.config.gamma / (self.config.v_std **2)
        gamma_wz = self.config.gamma / (self.config.w_std **2)

        # Control cost: penalize large devitions from nominal
        dv = v_samples - self.control_sequence.vx.unsqueeze(0) #[K, T]
        dw = w_samples - self.control_sequence.wz.unsqueeze(0) #[K, T]
        # control_cost = gamma_vx * (dv*dv).sum(dim=1) + gamma_wz * (dw*dw).sum(dim=1)
        # control_cost = (gamma_vx * (dv*dv).sum(dim=1) + 
        #            gamma_wz * (dw*dw).sum(dim=1)) / T
        # Nominal controls
        vx_nom = self.control_sequence.vx.unsqueeze(0)  # [1, T]
        wz_nom = self.control_sequence.wz.unsqueeze(0)  # [1, T]
        
        # MPPI's asymmetric control cost (matches paper)
        control_cost = (gamma_vx * (dv * vx_nom).sum(dim=1) +
                    gamma_wz * (dw * wz_nom).sum(dim=1)) 
        total_costs = total_costs + control_cost


        # Compute importance weights (softmax)
        min_cost = torch.min(total_costs)
        weights = torch.exp(-(total_costs - min_cost) / self.config.lambda_)
        weights = weights / torch.sum(weights)  # Normalize
        
        
        # Weighted Average along Control Samples
        # High note we have 1000 56 values of we have weights which are sum along over respected horizon
        # Now those [K] taken broadcasted at [K, 1] ~ converted to cols then whe we multiply with v sampled 
        # we have weighted velocity along all trajectory and horizon now we take sum along all trajectory at repected t
        v_new = torch.einsum('k,kt->t', weights, v_samples)  # [T]
        w_new = torch.einsum('k,kt->t', weights, w_samples)  # [T]

        # Update control sequence
        self.control_sequence.vx = v_new
        self.control_sequence.wz = w_new

        # Smoothing #
        # self._apply_smoothing(window_size=5)
        self.control_sequence.clamp(
            self.config.v_min,
            self.config.v_max,
            self.config.w_max
        )

        v, w = self.control_sequence.get_first_command()

        self.control_sequence.shift()   # warm_start for next iteration

        # if self._debug_counter % 50 == 0:
        #     # progress_info = self.path_tracker.get_progress_info()
        #     print(f"\n🎮 MPPI Control (Cycle {self._debug_counter}):")
        #     print(f"   Progress: {progress_info['progress_pct']:.1f}%")
        #     print(f"   Remaining: {progress_info['remaining_distance']:.2f}m")
        #     print(f"   Command: v={v:.3f} m/s, ω={w:.3f} rad/s")
        #     print(f"   Cost range: [{total_costs.min().item():.1f}, "
        #           f"{total_costs.max().item():.1f}]")
            
        self._debug_counter += 1
        return v, w
    
    # def get_progress(self) -> dict:
    #     """
    #     Get current progress along the path.
        
    #     Returns:
    #         Dictionary with progress information
    #     """
    #     return self.path_tracker.get_progress_info()
    
    def reset(
        self,
        new_path: np.ndarray = None,
        new_goal: np.ndarray = None
    ):
        """
        Reset controller for new goal/path.
        
        Args:
            new_path: [N, 2] or [N, 3] new path (optional)
            new_goal: [2] new goal (optional)
        """
        if new_path is not None and new_goal is not None:
            self.path_tracker.reset(new_path, new_goal)
        
        # Reset control sequence
        self.control_sequence.initalize_with_forward_motion(v=0.4)
        
        # Reset statistics
        if self.critic_manager.publish_stats:
            self.critic_manager.reset_statistics()
        
        self._debug_counter = 0
        
        print(" MPPIController reset")



    



