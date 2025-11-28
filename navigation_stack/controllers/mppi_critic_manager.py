# navigation_stack/controllers/mppi_critic_manager.py

import torch
from typing import Optional
from .mppi_costs import (
    PathTrackingCost,
    PathAngleCritic,
    PathFollowCritic,
    CostCritic,
    ObstaclesCritic,
    GoalCritic,
    GoalAngleCritic,
    ConstraintCritic,
    SmoothnessCritic,
    TwirlingCritic,
    PreferForwardCritic,
    DeadbandCritic
)

class CriticManager:
    """
    Manage all MPPI critics and combines their costs

    This is central hub that:
    1. Initializes all critics with weights
    2. Calls each critic's compute() method
    3. Sums up all costs for trajectory ranking

    Usage: 
        manager = CriticManager(config)
        total_costs = manager.evaluate_trajectories(
            trajectories, v_samples, w_samples,
            path, goal, current_pose, costmap, grid_origin, dt
        )
    """
    def __init__(
        self,
        # Obstacle avoidance weights
        cost_critic_weight: float = 50.0,
        obstacles_critic_weight: float = 50.0,
        
        # Path following weights
        path_tracking_weight: float = 10.0,
        path_angle_weight: float = 15.0,
        path_follow_weight: float = 12.0,
        
        # Goal reaching weights
        goal_weight: float = 15.0,
        goal_angle_weight: float = 8.0,
        
        # Motion quality weights
        constraint_weight: float = 10.0,
        smoothness_weight: float = 10.0,
        twirling_weight: float = 15.0,
        prefer_forward_weight: float = 15.0,
        
        # Optional critic
        use_deadband_critic: bool = False,
        deadband_weight: float = 5.0,
        
        # Critic parameters
        resolution: float = 0.1,
        v_max: float = 0.5,
        v_min: float = -0.1,
        w_max: float = 1.0,
        
        # Path/Goal thresholds
        goal_distance_threshold: float = 2.0,
        goal_angle_threshold: float = 0.8,
        twirling_threshold: float = 0.5,
        
        # Optional: Fixed goal heading
        goal_heading: Optional[float] = None):
         
        """
        Initialize all critics with their weights and parameters.
        
        Args:
            *_weight: Weight multipliers for each critic (higher = more important)
            use_deadband_critic: Whether to use optional deadband critic
            resolution: Costmap resolution (meters per cell)
            v_max, v_min, w_max: Velocity constraints
            goal_distance_threshold: Distance to activate GoalCritic
            goal_angle_threshold: Distance to activate GoalAngleCritic
            twirling_threshold: Distance to activate TwirlingCritic
            goal_heading: Optional fixed goal heading (radians)
        """
        # Obstacle Avoidance Critics:
        self.cost_critic = CostCritic(
            weight=cost_critic_weight,
            critical_threshold=0.8,
            lethal_threshold=0.95,
            critical_cost=300.0,
            lethal_cost=1e6,
            stride=5,
            resolution=resolution
        )
        
        self.obstacles_critic = ObstaclesCritic(
            weight=obstacles_critic_weight,
            critical_threshold=0.8,
            lethal_threshold=0.95,
            critical_cost=300.0,
            collision_cost=1e6,
            stride=5,
            resolution=resolution,
            repulsion_weight=2.0
        )

        # Path following critics

        self.path_tracking_critic = PathTrackingCost(
            weight=path_tracking_weight
        )
        
        self.path_angle_critic = PathAngleCritic(
            weight=path_angle_weight,
            offset_from_furthest=15,
            max_angle_to_furthest=1.57,  # ~90 degrees
            stride=5
        )
        
        self.path_follow_critic = PathFollowCritic(
            weight=path_follow_weight,
            offset_from_furthest=20
        )

        # Goal reaching Critics
        self.goal_critic = GoalCritic(
            weight=goal_weight,
            threshold_to_consider=goal_distance_threshold
        )
        
        self.goal_angle_critic = GoalAngleCritic(
            weight=goal_angle_weight,
            threshold_to_consider=goal_angle_threshold,
            goal_heading=goal_heading
        )

        # Motion Quality Critics
        self.constraint_critic = ConstraintCritic(
            weight=constraint_weight,
            v_max=v_max,
            v_min=v_min,
            w_max=w_max
        )
        
        self.smoothness_critic = SmoothnessCritic(
            weight=smoothness_weight,
            v_weight=1.0,
            w_weight=1.0
        )
        
        self.twirling_critic = TwirlingCritic(
            weight=twirling_weight,
            goal_dist_threshold=twirling_threshold
        )
        
        self.prefer_forward_critic = PreferForwardCritic(
            weight=prefer_forward_weight,
            threshold=0.05
        )

        # Deadband Critics

        self.use_deadband = use_deadband_critic
        if use_deadband_critic:
            self.deadband_critic = DeadbandCritic(
                weight=deadband_weight,
                v_deadband=0.03,
                w_deadband=0.05
            )
        else:
            self.deadband_critic = None

    def evaluate_trajectories(
        self,
        trajectories: torch.Tensor, # [K, T, 3] - Sample Trajectories
        v_samples: torch.Tensor,    # [K, T] - velocity samples
        w_samples: torch.Tensor,    # [K, T] - angular velocity samples
        path: torch.Tensor,         # [P, 2] - global path (XY only)
        goal: torch.Tensor,         # [2] - final goal position
        current_pose: torch.Tensor, #[3]- current robot pose (X, Y, Theta)
        costmap: torch.Tensor,      #[H, W] - STVL costmap (0-1)
        grid_origin: torch.Tensor,  #[2] - costmap origin (robot-centric)
        dt: float                   # Timestep
    )-> torch.Tensor:
        """
        Evaluate all critics and return total cost for each trajectory.
        
        Args:
            trajectories: K sampled trajectories over T timesteps
            v_samples: Linear velocity samples
            w_samples: Angular velocity samples
            path: Global path waypoints (XY coordinates only)
            goal: Final goal position (XY)
            current_pose: Current robot state (x, y, theta)
            costmap: Robot-centric STVL costmap
            grid_origin: Costmap origin in world frame
            dt: Timestep duration
        
        Returns:
            total_costs: [K] - Total cost for each trajectory
                        (Lower cost = better trajectory)
        """
        K = trajectories.shape[0]
        device = trajectories.device
        
        # Initialize total costs
        total_costs = torch.zeros(K, device=device)

        #Obstacle avoidance 
        # Direct costmap lookup
        total_costs += self.cost_critic.compute(
            trajectories, costmap, grid_origin
        )
        
        # Distance-based obstacle cost
        total_costs += self.obstacles_critic.compute(
            trajectories, costmap, grid_origin
        )

        #Path Following 
        # Stay close to path
        total_costs += self.path_tracking_critic.compute(
            trajectories, path
        )
        
        # Point toward path
        total_costs += self.path_angle_critic.compute(
            trajectories, path, current_pose
        )
        
        # Drive toward specific path point
        total_costs += self.path_follow_critic.compute(
            trajectories, path
        )

        # Goal Reaching

        # Attract to goal position
        total_costs += self.goal_critic.compute(
            trajectories, goal, current_pose
        )
        
        # Align with goal heading
        total_costs += self.goal_angle_critic.compute(
            trajectories, goal, current_pose, path=path
        )

        #Motion Quality
        # Enforce velocity limits
        total_costs += self.constraint_critic.compute(
            v_samples, w_samples
        )
        
        # Smooth motion (reduce jerk)
        total_costs += self.smoothness_critic.compute(
            v_samples, w_samples
        )
        
        # Reduce excessive rotation
        total_costs += self.twirling_critic.compute(
            w_samples, goal, current_pose
        )
        
        # Prefer forward motion
        total_costs += self.prefer_forward_critic.compute(
            v_samples
        )

        # Deadband 

        if self.use_deadband:
            total_costs += self.deadband_critic.compute(
                v_samples, w_samples
            )
        
        return total_costs
    
    def get_weights_summary(self) -> dict:
        """
        Get current critic weights for debugging/tuning.
        
        Returns:
            Dictionary of critic names and their weights
        """
        weights = {
            "obstacle_avoidance": {
                "cost_critic": self.cost_critic.weight,
                "obstacles_critic": self.obstacles_critic.weight,
            },
            "path_following": {
                "path_tracking": self.path_tracking_critic.weight,
                "path_angle": self.path_angle_critic.weight,
                "path_follow": self.path_follow_critic.weight,
            },
            "goal_reaching": {
                "goal": self.goal_critic.weight,
                "goal_angle": self.goal_angle_critic.weight,
            },
            "motion_quality": {
                "constraint": self.constraint_critic.weight,
                "smoothness": self.smoothness_critic.weight,
                "twirling": self.twirling_critic.weight,
                "prefer_forward": self.prefer_forward_critic.weight,
            }
        }
        
        if self.use_deadband:
            weights["motion_quality"]["deadband"] = self.deadband_critic.weight
        
        return weights
    
    def update_weight(self, critic_name: str, new_weight: float):
        """
        Update a critic's weight dynamically (for tuning).
        
        Args:
            critic_name: Name of the critic (e.g., "path_tracking", "obstacles")
            new_weight: New weight value
        
        Example:
            manager.update_weight("twirling", 30.0)  # Increase twirling penalty
        """
        critic_map = {
            "cost": self.cost_critic,
            "obstacles": self.obstacles_critic,
            "path_tracking": self.path_tracking_critic,
            "path_angle": self.path_angle_critic,
            "path_follow": self.path_follow_critic,
            "goal": self.goal_critic,
            "goal_angle": self.goal_angle_critic,
            "constraint": self.constraint_critic,
            "smoothness": self.smoothness_critic,
            "twirling": self.twirling_critic,
            "prefer_forward": self.prefer_forward_critic,
        }
        
        if self.use_deadband:
            critic_map["deadband"] = self.deadband_critic
        
        if critic_name in critic_map:
            critic_map[critic_name].weight = new_weight
            print(f"Updated {critic_name} weight to {new_weight}")
        else:
            print(f"Unknown critic: {critic_name}")
            print(f"Available: {list(critic_map.keys())}")