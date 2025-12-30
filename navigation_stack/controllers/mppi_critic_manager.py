# navigation_stack/controllers/mppi_critic_manager.py

import torch
from typing import List, Optional, Dict
from .critic_data import CriticData
from .mppi_costs import (
    PathAlignCritic,
    PathAngleCritic,
    PathFollowCritic,
    CostCritic,
    GoalCritic,
    GoalAngleCritic,
    ConstraintCritic,
    TwirlingCritic,
    PreferForwardCritic,
    DeadbandCritic,
    SpeedIncentiveCritic
)


class CriticManager:
    """
    Manage all MPPI critics and combines their costs

    This is the central hub that:
    1. Initializes all critics with their weights
    2. Prepares unified CriticData structure
    3. Calls each critic's compute() method
    4. Tracks statistics (optional)
    5. Sums up costs for trajectory ranking

    Usage:
        # Initialize
        manager = CriticManager(config)
        
        # Every MPPI cycle:
        data = manager.prepare_critic_data(
            trajectories, v_samples, w_samples,
            path_tracker, costmap, grid_origin, ...
        )
        total_costs = manager.evaluate_trajectories(data)
    
    Key Features:
    - Unified CriticData structure (matches Nav2)
    - Lazy path validity computation (only when needed)
    - Optional statistics tracking
    - Early exit on fail_flag
    """
    def __init__(
        self,
        # Obstacle avoidance weights
        cost_critic_weight: float = 4.0,
        
        # Path following weights
        path_align_weight: float = 10.0,
        path_angle_weight: float = 2.2,
        path_follow_weight: float = 5.0,
        
        # Goal reaching weights
        goal_weight: float = 5.0,
        goal_angle_weight: float = 3.0,
        
        # Motion quality weights
        constraint_weight: float = 3.0,
        # smoothness_weight: float = 5.0,
        twirling_weight: float = 1.5,
        prefer_forward_weight: float = 5.0,
        
        # Optional critics
        use_deadband_critic: bool = False,
        deadband_weight: float = 35.0,
        
        # Critic parameters
        resolution: float = 0.1,
        speed_incentive_weight: float = -5.0,  # NEGATIVE!
        enable_speed_incentive: bool = True,
        # Enable/disable critics
        enable_cost_critic: bool = True,
        enable_path_align: bool = True,
        enable_path_angle: bool = True,
        enable_path_follow: bool = True,
        enable_goal: bool = True,
        enable_goal_angle: bool = True,
        enable_constraint: bool = True,
        # enable_smoothness: bool = True,
        enable_twirling: bool = True,
        enable_prefer_forward: bool = True,
        
        # Statistics
        publish_stats: bool = False,
        verbose: bool = False
        ):
        """
        Initialize CriticManager with all critics.
        
        Args:
            *_weight: Weight multipliers for each critic
            use_deadband_critic: Whether to use deadband critic
            resolution: Costmap resolution (meters per cell)
            enable_*: Enable/disable individual critics
            publish_stats: Track and print statistics
            verbose: Print detailed debug info
        """
        self.resolution = resolution
        self.publish_stats = publish_stats
        self.verbose = verbose

        # Statistics tracking
        self.stats_history: List[Dict] = []
        self._cycle_count = 0

        self.critics = []  # List of active critics
        self.critic_names = []  # Names for debugging
        
        # Obstacle Avoidance

        if enable_cost_critic and cost_critic_weight > 0:
            self.critics.append(CostCritic(
                weight=cost_critic_weight,
                resolution=resolution
            ))
            self.critic_names.append("CostCritic")

        # Path Following
        if enable_path_align and path_align_weight > 0:
            self.critics.append(PathAlignCritic(weight=path_align_weight))
            self.critic_names.append("PathAlignCritic")
        
        if enable_path_angle and path_angle_weight > 0:
            self.critics.append(PathAngleCritic(weight=path_angle_weight))
            self.critic_names.append("PathAngleCritic")
        
        if enable_path_follow and path_follow_weight > 0:
            self.critics.append(PathFollowCritic(weight=path_follow_weight))
            self.critic_names.append("PathFollowCritic")

        # Goal Reaching
        if enable_goal and goal_weight > 0:
            self.critics.append(GoalCritic(weight=goal_weight))
            self.critic_names.append("GoalCritic")
        
        if enable_goal_angle and goal_angle_weight > 0:
            self.critics.append(GoalAngleCritic(weight=goal_angle_weight))
            self.critic_names.append("GoalAngleCritic")
        
        # Motion Quality
        if enable_constraint and constraint_weight > 0:
            self.critics.append(ConstraintCritic(weight=constraint_weight))
            self.critic_names.append("ConstraintCritic")
        
        # if enable_smoothness and smoothness_weight > 0:
        #     self.critics.append(SmoothnessCritic(weight=smoothness_weight))
        #     self.critic_names.append("SmoothnessCritic")
        
        if enable_twirling and twirling_weight > 0:
            self.critics.append(TwirlingCritic(weight=twirling_weight))
            self.critic_names.append("TwirlingCritic")
        
        if enable_prefer_forward and prefer_forward_weight > 0:
            self.critics.append(PreferForwardCritic(weight=prefer_forward_weight))
            self.critic_names.append("PreferForwardCritic")
        
        # Optional Deadband
        if use_deadband_critic and deadband_weight > 0:
            self.critics.append(DeadbandCritic(weight=deadband_weight))
            self.critic_names.append("DeadbandCritic")

        if enable_speed_incentive and speed_incentive_weight != 0.0:
            self.critics.append(SpeedIncentiveCritic(
                weight=speed_incentive_weight
            ))
            self.critic_names.append("SpeedIncentiveCritic")


        print(f" CriticManager initialized with {len(self.critics)} critics:")
        for name in self.critic_names:
            print(f"   - {name}") 

    def prepare_critic_data(
        self,
        trajectories: torch.Tensor,
        v_samples: torch.Tensor,
        w_samples: torch.Tensor,
        path_tracker,  # PathTracker instance
        costmap: torch.Tensor,
        grid_origin: torch.Tensor,
        current_pose: torch.Tensor,
        dt: float,
        v_max: float,
        v_min: float,
        w_max: float,
        compute_path_validity: bool = True
    ) -> CriticData:
        """
        Prepare unified CriticData structure for all critics.
        
        This is called ONCE per MPPI optimization cycle.
        
        Args:
            trajectories: [K, T, 3] sampled trajectories
            v_samples: [K, T] linear velocity samples
            w_samples: [K, T] angular velocity samples
            path_tracker: PathTracker instance (contains path state)
            costmap: [H, W] STVL costmap
            grid_origin: [2] costmap origin
            current_pose: [3] robot pose (x, y, theta)
            dt: Timestep
            v_max, v_min, w_max: Velocity limits
            compute_path_validity: Whether to compute path validity
        
        Returns:
            CriticData: Unified data structure for all critics
        """
        # Compute path validity (ONCE per cycle, not every frame!)
        # This is the answer to your question!
        if compute_path_validity:
            path_tracker.compute_path_validity(
                costmap=costmap,
                grid_origin=grid_origin,
                resolution=self.resolution
            )
        
        tracker_data = path_tracker.get_critic_data()
   
        
        # Create CriticData
        return CriticData(
            trajectories=trajectories,
            v_samples=v_samples,
            w_samples=w_samples,

            path=tracker_data['pruned_path'],  
            path_integrated_distances=tracker_data['path_integrated_distances'],
            path_valid_flags=tracker_data['path_valid_flags'],
            
            goal=tracker_data['goal'],
            goal_heading=tracker_data['goal_heading'],
            # furthest_reached_idx=0,  # Correct: it is 0 relative to the pruned path
            local_path_length=tracker_data['local_path_length'],
            
            current_pose=current_pose,
            costmap=costmap,
            grid_origin=grid_origin,
            dt=dt,
            v_max=v_max,
            v_min=v_min,
            w_max=w_max
        )



    def evaluate_trajectories(self, data: CriticData) -> torch.Tensor:
        """
        Evaluate all critics and return total cost.
        
        Matches Nav2's evalTrajectoriesScores() function.
        
        Args:
            data: Unified CriticData structure
        
        Returns:
            total_costs: [K] Total cost for each trajectory
        """
        K = data.trajectories.shape[0]
        device = data.trajectories.device
        
        # Initialize costs
        total_costs = torch.zeros(K, device=device)
        
        # Track statistics if enabled
        stats = {}
        if self.publish_stats:
            stats['cycle'] = self._cycle_count
            stats['critics'] = []
            stats['costs_before'] = []
            stats['costs_after'] = []
            stats['costs_added'] = []
        
        # Evaluate each critic
        fail_flag = False
        for i, (critic, name) in enumerate(zip(self.critics, self.critic_names)):
            # Early exit on failure (matches Nav2)
            if fail_flag:
                if self.verbose:
                    print(f"⚠️  Early exit: fail_flag set by previous critic")
                break
            
            # Store costs before (for statistics)
            if self.publish_stats:
                costs_before = total_costs.clone()
            
            # Compute critic cost
            try:
                critic_cost = critic.compute(data)
                total_costs += critic_cost
                
                # Check for failures (all trajectories in collision)
                # This is set by CostCritic if all trajectories collide
                if hasattr(data, 'fail_flag'):
                    fail_flag = data.fail_flag
                
            except Exception as e:
                print(f"❌ Critic {name} failed: {e}")
                # Continue with other critics
                continue
            
            # Track statistics
            if self.publish_stats:
                costs_after = total_costs.clone()
                costs_added = costs_after - costs_before
                
                stats['critics'].append(name)
                stats['costs_before'].append(costs_before.mean().item())
                stats['costs_after'].append(costs_after.mean().item())
                stats['costs_added'].append(costs_added.sum().item())
        
        # Store statistics
        if self.publish_stats:
            self.stats_history.append(stats)
            
            # Print every N cycles
            if self._cycle_count % 50 == 0:
                self.print_statistics(stats)
        
        self._cycle_count += 1
        
        return total_costs
    
    def print_statistics(self, stats: Dict):
        """Print critic statistics (matches Nav2's stats publishing)."""
        print(f"\n📊 Critic Statistics (Cycle {stats['cycle']}):")
        print(f"{'Critic':<20} {'Cost Added':>15} {'Mean Before':>15} {'Mean After':>15}")
        print("-" * 70)
        
        for i, name in enumerate(stats['critics']):
            cost_added = stats['costs_added'][i]
            mean_before = stats['costs_before'][i]
            mean_after = stats['costs_after'][i]
            
            # Highlight significant contributions
            marker = "🔥" if abs(cost_added) > 100 else "  "
            
            print(f"{marker} {name:<18} {cost_added:>15.2f} "
                  f"{mean_before:>15.2f} {mean_after:>15.2f}")
        
        print("-" * 70)
    
    def get_weights_summary(self) -> Dict[str, float]:
        """Get current critic weights for debugging/tuning."""
        weights = {}
        for critic, name in zip(self.critics, self.critic_names):
            weights[name] = critic.weight
        return weights
    
    def update_weight(self, critic_name: str, new_weight: float):
        """
        Update a critic's weight dynamically (for tuning).
        
        Args:
            critic_name: Name of critic (e.g., "PathAlignCritic")
            new_weight: New weight value
        """
        for critic, name in zip(self.critics, self.critic_names):
            if name == critic_name:
                critic.weight = new_weight
                print(f"✅ Updated {critic_name} weight: {new_weight}")
                return
        
        print(f"❌ Unknown critic: {critic_name}")
        print(f"Available: {self.critic_names}")
    
    def reset_statistics(self):
        """Reset statistics tracking."""
        self.stats_history.clear()
        self._cycle_count = 0
    
    def get_statistics_summary(self) -> Dict:
        """Get summary of all tracked statistics."""
        if not self.stats_history:
            return {}
        
        summary = {
            'total_cycles': len(self.stats_history),
            'critics': self.critic_names,
            'average_costs': {}
        }
        
        # Compute averages
        for name in self.critic_names:
            costs = []
            for stats in self.stats_history:
                if name in stats['critics']:
                    idx = stats['critics'].index(name)
                    costs.append(stats['costs_added'][idx])
            
            if costs:
                summary['average_costs'][name] = sum(costs) / len(costs)
        
        return summary