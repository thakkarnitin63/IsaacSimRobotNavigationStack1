# In: navigation_stack/controllers/mppi_costs.py
import torch
from typing import Optional

class PathTrackingCost:
    """
    Simplest Possble cost: distance to closest path point.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def compute(
        self,
        trajectories: torch.Tensor, # [K, T, 3]
        path: torch.Tensor,         #[P, 2]
    )-> torch.Tensor:
        """
        Returns: costs [K] (one scalar cost per trajectory)
        """
        K, T, _ = trajectories.shape

        # Extract XY position: [K, T, 2]
        positions = trajectories[:, :, :2]

        # Reshape for distance computation: [K*T ,2] [56000, 2]
        positions_flat = positions.reshape(K*T, 2)

        # Compute distance to All path points: [K*T, P][56000 , avg 200]
        distances = torch.cdist(positions_flat, path)

        # Find minimum distance for each position: [K*T][row along 56000 col along path]
        min_distances, _ = torch.min(distances, dim=1)

        # Reshape back and average over time: [K, T]-> [K]
        min_distances = min_distances.reshape(K, T) #[56000 -> minimum value so each point is how far from trajectory at ref standpoint]
        avg_distance = torch.mean(min_distances, dim=1)


        return self.weight*avg_distance 
    
class CostCritic:
    """
        Direct Costmap Lookup for STVL costmap (values 0.0 to 1.0)
        Key: Costmap origin moves with robot! 
        grid_origin = robot_pose + robot_centric_offset

        Interprets costmap values:
        - 0.0-0.2: Free space (minimal cost)
        - 0.2-0.5: Approaching obstacles (moderate cost)
        - 0.5-0.8: Close to obstacles (high cost)
        - 0.8-0.95: Very close (critical cost)
        - 0.95-1.0: Collision (lethal cost)
    """
    def __init__(
        self,
        weight: float = 50.0,
        critical_threshold: float = 0.8, # Critical Zone starts here
        lethal_threshold: float  = 0.95,  # Lethal Zone starts here
        critical_cost: float = 300.0,
        lethal_cost: float = 1e6,
        stride: int = 5, # Check every 5th trajectory point
        resolution: float = 0.1 # meters per cell
        ):
        self.weight = weight
        self.critical_threshold = critical_threshold
        self.lethal_threshold = lethal_threshold
        self.critical_cost = critical_cost
        self.lethal_cost = lethal_cost
        self.stride = stride
        self.resolution = resolution

    def compute(
        self,
        trajectories: torch.Tensor, # [K, T, 3]
        costmap: torch.Tensor,      # [H, W] with values [0, 1]
        grid_origin: torch.Tensor   # [2] current grid origin (moves with robot!)
    )-> torch.Tensor:
        """
        Args: 
            grid_origin: Current grid origin in world frame
                        = robot_pose_2d + robot_centric_offset
        """
        K, T, _ = trajectories.shape
        H, W = costmap.shape # can be 128*128 or 256*256

        # Sample trajectory points
        positions = trajectories[:, ::self.stride, :2] # [K, T',  2] T' every 5th value
        T_sampled = positions.shape[1]

        # Convert world coordinates to grid coordinates
        # (world_x - grid_origin[0]) / voxel_size for indexing

        grid_x = ((positions[:, : , 0] - grid_origin[0]) / self.resolution).long()
        grid_y = ((positions[:, : , 1] - grid_origin[1]) / self.resolution).long()

        # Clamp to valid range [0 , W-1] and [0, H-1] 
        grid_x = torch.clamp(grid_x, 0, W-1)
        grid_y = torch.clamp(grid_y, 0, H-1)

        #Look up costs
        costs = costmap[grid_x, grid_y] # [K, T'] with values [0, 1]

        # Three tier penalty based on STVL cost values
        penalties = torch.zeros_like(costs, dtype=torch.float32)

        # Lethal Zone (0.95 - 1.0)
        lethal_mask = costs >= self.lethal_threshold
        penalties[lethal_mask] = self.lethal_cost

        # Critical Zone (0.8 - 0.95)
        critical_mask = (costs>= self.critical_threshold) & (costs < self.lethal_threshold)
        penalties[critical_mask] = self.critical_cost

        # Repulsive zone (0.2 - 0.8): Exponential increase
        repulsive_mask = (costs >= 0.2) & (costs < self.critical_threshold)
        normalized_cost = (costs[repulsive_mask] -0.2) / 0.6     #[0, 1]
        penalties[repulsive_mask] = 50.0 * torch.exp(3.0 * normalized_cost)

        # Free space (0.0 - 0.2): Minimal cost
        free_mask = costs < 0.2
        penalties[free_mask] = costs[free_mask] * 10.0

        # Sum over trajectory
        total_cost = torch.sum(penalties, dim=1) #[K]

        return self.weight * total_cost

    
class ObstaclesCritic:
    """
    Distance-based obstacle cost for Robot Centric STVL costmap (0.0 to 1.0).
    Key: Costmap origin moves with robot!

    """
    def __init__(
        self,
        weight: float = 50.0,
        critical_threshold: float = 0.8,
        lethal_threshold: float = 0.95,
        critical_cost: float = 300.0,
        collision_cost: float = 1e6,
        stride: int = 5,
        resolution: float = 0.1,
        repulsion_weight: float = 2.0
    ):
        self.weight = weight
        self.critical_threshold = critical_threshold
        self.lethal_threshold = lethal_threshold
        self.critical_cost = critical_cost
        self.collision_cost = collision_cost
        self.stride = stride
        self.resolution = resolution
        self.repulsion_weight = repulsion_weight

    def compute(
        self,
        trajectories: torch.Tensor,
        costmap: torch.Tensor,
        grid_origin: torch.Tensor  # [2] current grid origin
    ) -> torch.Tensor:
        K, T, _ = trajectories.shape
        H, W = costmap.shape
        
        # Sample positions
        positions = trajectories[:, ::self.stride, :2]
        T_sampled = positions.shape[1]
        
        # Grid coordinates (robot-centric!)
        grid_x = ((positions[:, :, 0] - grid_origin[0]) / self.resolution).long()
        grid_y = ((positions[:, :, 1] - grid_origin[1]) / self.resolution).long()
        grid_x = torch.clamp(grid_x, 0, W - 1)
        grid_y = torch.clamp(grid_y, 0, H - 1)
        
        # Lookup costs
        costs = costmap[grid_x, grid_y].float()  # [K, T'] ~[0, 1]
        
        penalties = torch.zeros_like(costs)
        
        # Collision zone (>=0.95)
        collision_mask = costs >= self.lethal_threshold
        penalties[collision_mask] = self.collision_cost
        
        # Critical zone (0.8-0.95)
        critical_mask = (costs >= self.critical_threshold) & (costs < self.lethal_threshold)
        penalties[critical_mask] = self.critical_cost
        
        # Repulsion zone (0.2-0.8): Exponential increase
        repulsion_mask = (costs >= 0.2) & (costs < self.critical_threshold)
        penalties[repulsion_mask] = self.repulsion_weight * torch.exp(
            4.0 * costs[repulsion_mask]
        )
        
        # Free space (0.0-0.2): Very small cost
        free_mask = costs < 0.2
        penalties[free_mask] = costs[free_mask] * 5.0
        
        # Sum over trajectory
        total_cost = torch.sum(penalties, dim=1)  # [K]
        
        return self.weight * total_cost
    

class PathAngleCritic:
    """
    Ensures robot points toward next path point.
    Computes desired heading on the fly from path geometry
    """
    def __init__(
        self,
        weight: float = 5.0,
        offset_from_furthest: int = 4,
        max_angle_to_furthest: float = 1.57,
        stride: int = 5
    ):
        self.weight = weight
        self.offset_from_furthest = offset_from_furthest
        self.max_angle_to_furthest = max_angle_to_furthest
        self.stride = stride

    def compute(
        self,
        trajectories: torch.Tensor,
        path: torch.Tensor, # potential bug
        current_pose: torch.Tensor
    )-> torch.Tensor:
        
        K, T, _ = trajectories.shape

        target_idx = min(self.offset_from_furthest, len(path) - 1)
        target_xy = path[target_idx]  #[2]

        sampled_positions = trajectories[:, ::self.stride, :2]
        sampled_headings = trajectories[: , ::self.stride, 2]

        dx = target_xy[0] - sampled_positions[: ,:, 0]
        dy = target_xy[1] - sampled_positions[:, :, 1]
        desired_headings = torch.atan2(dy,dx)

        angle_errors = torch.atan2( # read math later
            torch.sin(desired_headings - sampled_headings),
            torch.cos(desired_headings - sampled_headings)
        )

        angle_errors = torch.clamp(
            torch.abs(angle_errors),
            0,
            self.max_angle_to_furthest
        )

        mean_error = torch.mean(angle_errors, dim=1) # across all 56 timestep so # [K]

        return self.weight * mean_error
    

class PathFollowCritic:
    """
    Drives Trajectory endpoint toward specific path point
    """
    def __init__(
        self,
        weight: float = 8.0,
        offset_from_furthest: int = 6,
        ):
        self.weight = weight
        self.offset_from_furthest = offset_from_furthest

    def compute(
        self,
        trajectories: torch.Tensor,
        path: torch.Tensor
    )-> torch.Tensor:
        target_idx = min(self.offset_from_furthest, len(path) - 1)
        target_xy = path[target_idx]

        final_positions = trajectories[:, -1, :2]
        # distance between final position last step #[1000 2 - 6th index in given path (single xy)]
        # diff tensor (1000, 2) distance along x y from ref
        # dim=1 take value from xy no if dim=0 then all respective x y would have all x and all y value but we want
        # specific distance so collapse dim=1 along x and y to find norm sqrt(x**2 + y**2) will have [1000]
        distance = torch.norm(final_positions - target_xy, dim=1)
        
        return self.weight * distance 

class GoalCritic:
    """
    Attracts robot to final goal position.
    """
    def __init__(
        self,
        weight: float = 15.0,
        threshold_to_consider: float = 2.0   
    ):
        self.weight = weight
        self.threshold_to_consider = threshold_to_consider

    def compute(
        self,
        trajectories: torch.Tensor,
        goal: torch.Tensor,
        current_pose: torch.Tensor
    )-> torch.Tensor:
        
        dist_to_goal = torch.norm(current_pose[:2]-goal)
        if dist_to_goal > self.threshold_to_consider:
            return torch.zeros(trajectories.shape[0], device= trajectories.device)
        
        final_positions = trajectories[:, -1, :2]
        distance = torch.norm(final_positions - goal, dim=1)

        return self.weight * distance
    
class GoalAngleCritic:
    """
    Aligns final heading with goal direction.
    """
    def __init__(
        self,
        weight: float = 8.0,
        threshold_to_consider: float = 0.8,
        goal_heading: Optional[float] = None
        ):
        self.weight = weight
        self.threshold_to_consider = threshold_to_consider
        self.goal_heading = goal_heading
    
    def compute(
        self,
        trajectories: torch.Tensor,
        goal: torch.Tensor,
        current_pose: torch.Tensor,
        path: Optional[torch.Tensor] = None
    )-> torch.Tensor:
        dist_to_goal = torch.norm(current_pose[:2]- goal)
        if dist_to_goal > self.threshold_to_consider:
            return torch.zeros(trajectories.shape[0], device=trajectories.device)
        
        final_heading = trajectories[:, -1, 2]
        
        if self.goal_heading is not None:
            target_heading = torch.tensor(
                self.goal_heading,
                device=trajectories.device  
            )
        elif path is not None and len(path)>= 2:
            dx = path[-1, 0] - path[-2, 0]
            dy = path[-1, 1] - path[-2, 1]
            target_heading = torch.atan2(dy,dx)
        else:
            return torch.zeros(trajectories.shape[0], device=trajectories.device)

        angle_errors = torch.abs(
            torch.atan2(
            torch.sin(target_heading - final_heading),
            torch.cos(target_heading - final_heading)
            )
        )

        return self.weight * angle_errors
    
class ConstraintCritic:
    """
    Enforces Velocity Limits(soft constraints)
    """
    def __init__(
        self,
        weight: float = 10.0,
        v_max: float = 0.5,
        v_min: float = -0.1,
        w_max: float = 1.0,            
    ):
        self.weight = weight
        self.v_max = v_max
        self.v_min = v_min
        self.w_max = w_max

    def compute(
        self,
        v_samples: torch.Tensor,
        w_samples: torch.Tensor
    )-> torch.Tensor:
        v_violations = (
            torch.clamp(v_samples - self.v_max, min=0) +
            torch.clamp(self.v_min - v_samples, min=0)
        )

        w_violations = torch.clamp(torch.abs(w_samples) - self.w_max, min=0)
        total_violations  = torch.sum(v_violations + w_violations, dim=1)

        return self.weight * total_violations

class SmoothnessCritic:
    """
    Penalizes jerky control changes.
    """
    def __init__(
        self,
        weight: float = 5.0,
        v_weight: float = 1.0,
        w_weight: float = 1.0
    ):
        self.weight = weight
        self.v_weight = v_weight
        self.w_weight = w_weight

    def compute(
        self,
        v_samples: torch.Tensor,
        w_samples: torch.Tensor
    )-> torch.Tensor:
        v_diff = torch.diff(v_samples, dim=1)
        w_diff = torch.diff(w_samples, dim=1)

        v_smoothness = torch.sum(v_diff**2, dim=1)
        w_smoothness = torch.sum(w_diff**2, dim=1)

        total_smoothness= (
            self.v_weight * v_smoothness +
            self.w_weight * w_smoothness
        )

        return self.weight * total_smoothness
    
class TwirlingCritic:
    """
    Reduces excessive rotation.
    Critical for fixing swirling behavior!
    """
    def __init__(
        self,
        weight: float = 15.0,
        goal_dist_threshold: float = 0.5
    ):
        self.weight = weight
        self.goal_dist_threshold = goal_dist_threshold

    def compute(
        self,
        w_samples: torch.Tensor,
        goal: torch.Tensor,
        current_pose: torch.Tensor
    )-> torch.Tensor:
        dist_to_goal = torch.norm(current_pose[:2]- goal)

        if dist_to_goal < self.goal_dist_threshold:
            return torch.zeros(w_samples.shape[0], device=w_samples.device)
        
        total_rotation = torch.sum(torch.abs(w_samples), dim=1)

        return self.weight * total_rotation
    

class PreferForwardCritic:
    """
    Discourages backward motion.
    """
    def __init__(
        self,
        weight: float = 5.0,
        threshold: float = 0.05,
    ):
        self.weight = weight
        self.threshold = threshold

    def compute(
        self,
        v_samples: torch.Tensor,        
    )-> torch.Tensor:
        backward_mask = v_samples < -self.threshold
        penalties = torch.abs(v_samples) * backward_mask.float()

        total_penalty = torch.sum(penalties, dim=1)

        return self.weight * total_penalty
    
class DeadbandCritic:
    """
    OPTIONAL: Penalize very small velocites.
    """
    def __init__(
        self,
        weight: float = 5.0,
        v_deadband: float = 0.03,
        w_deadband: float = 0.05
    ):
        self.weight = weight
        self.v_deadband = v_deadband
        self.w_deadband = w_deadband
    
    def compute(
        self,
        v_samples: torch.Tensor,
        w_samples: torch.Tensor,
    ) -> torch.Tensor:
        
        v_in_deadband = (torch.abs(v_samples) < self.v_deadband).float()
        w_in_deadband = (torch.abs(w_samples) < self.w_deadband).float()

        total_penalty = torch.sum(v_in_deadband + w_in_deadband, dim=1)

        return self.weight * total_penalty
        
