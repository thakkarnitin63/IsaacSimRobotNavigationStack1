# In: navigation_stack/controllers/mppi_costs.py
import torch
from typing import Optional
import math

    
class CostCritic: #done
    """
    Direct Costmap Lookup for STVL costmap (values 0.0 to 1.0)
    - Samples every Nth trajectory point (stride)
    - Disables repulsion near goal (near_goal logic)
    - Uses raw costmap values (not fixed penalties!)
    - Breaks on collision (via masking)
    - Normalizes by number of sampled points
    
    Default parameters (Nav2):
    - weight: 3.81 / 254.0 ≈ 0.015 (normalized)
    - critical_cost: 300.0
    - collision_cost: 1e6
    - near_goal_distance: 0.5m
    - stride: 2
    - power: 1
    """
    def __init__(
        self,
        weight: float = 3.81,
        critical_cost: float = 300.0,
        near_collision_threshold: float = 0.996, # 253/254
        collision_threshold: float = 0.999, # Lethal (254/254)
        collision_cost: float = 1e6,
        near_goal_distance: float = 0.5,
        stride: int = 2,
        resolution: float = 0.1,
        power: int = 1
        ):
        # Normalize weight like Nav2
        self.weight = weight / 254.0
        self.critical_cost = critical_cost
        self.near_collision_threshold = near_collision_threshold
        self.collision_threshold = collision_threshold
        self.collision_cost = collision_cost
        self.near_goal_distance = near_goal_distance
        self.stride = stride
        self.resolution = resolution
        self.power = power
        self._debug_counter = 0

    def compute(
        self,
        data: 'CriticData'
    )-> torch.Tensor:
        """
        Compute costmap-based obstacle cost (FULLY VECTORIZED).
        
        Args:
            data: CriticData containing all necessary information
        
        Returns:
            costs: [K] - Cost for each trajectory
        """
        K = data.trajectories.shape[0]
        device = data.trajectories.device

        # Check if near goal (scalar)
        near_goal = data.local_path_length < self.near_goal_distance

        # sample trajectory points with stride
        positions = data.trajectories[:, ::self.stride, :2]  #[K, T', 2]
        T_sampled = positions.shape[1]

        # Convert world -> grid coordinates
        grid_x = ((positions[:, :, 0] - data.grid_origin[0]) / self.resolution).long()
        grid_y = ((positions[:, :, 1]- data.grid_origin[1]) / self.resolution).long()

        # Clamp to valid range
        dim_x, dim_y = data.costmap.shape
        grid_x = torch.clamp(grid_x, 0, dim_x-1)
        grid_y = torch.clamp(grid_y, 0, dim_y-1)

        # lookup costmap values - [K, T']
        pose_costs = data.costmap[grid_x, grid_y]

        # Handle "break on collision" with masking 
        # Find first collision index for each trajectory
        collision_mask = pose_costs >= self.collision_threshold # [K, T'] - True where collision 253/254

        # For each trajectory, find first collision index (or T_sampled if none)
        # cumsum gives us the cumulative count of collisions
        collision_cumsum = collision_mask.long().cumsum(dim=1)  # [K, T']

        # Mask: only process points Before first collision
        # (collision_cumsum == 0) means no collision has occured yet
        valid_mask = collision_cumsum == 0 #[K, T']

        # if trajectory has Any collision, mark it 
        has_collision = collision_mask.any(dim=1) #[K]

        # Initialize costs array
        traj_costs = torch.zeros(K, device=device)
        
        # Handle collisions cases
        #Trajectories with collisions get full collision cost
        traj_costs[has_collision] = self.collision_cost / float(T_sampled)

        # For non colliding trajectories, accumulate costs
        non_collision_mask = ~has_collision #[K]

        if non_collision_mask.any():
            # Get costs for non-colliding trajectories only
            costs_subset = pose_costs[non_collision_mask] #[K' , T']
            valid_subset = valid_mask[non_collision_mask] #[K' , T']

            # Critical zone penalty (>= 0.95)
            critical_mask = (costs_subset >= self.near_collision_threshold) & valid_subset
            critical_costs = critical_mask.float() * self.critical_cost #[K', T']

            # Repulsive zone (normal costs, only if Not near goal)
            if not near_goal:
                # Use raw costmap values (scaled to 0 - 254)
                # Only where cost >= 0.004 (significant) and valid
                repulsive_mask = (costs_subset >= 0.004) & valid_subset & ~critical_mask
                repulsive_costs = costs_subset * 254.0 * repulsive_mask.float() #[K', T']
            else:
                repulsive_costs = torch.zeros_like(costs_subset)

            # Sum costs over time dimension
            total_costs = critical_costs.sum(dim=1) + repulsive_costs.sum(dim=1) #[K']

            # Normalize by number of sampled points
            traj_costs[non_collision_mask] = total_costs / float(T_sampled)
        
        # Apply weight and power
        weighted_costs = traj_costs * self.weight
        if self.power >1:
            weighted_costs = weighted_costs ** self.power

        # Debug
        if self._debug_counter % 200 == 0:
            print(f"🟡 CostCritic ACTIVE")
            print(f"   Near goal: {near_goal}")
            print(f"   Sampled points: {T_sampled} (stride={self.stride})")
            print(f"   Collisions: {has_collision.sum().item()}/{K} trajectories")
            print(f"   Avg cost: {traj_costs.mean().item():.2f}")
            print(f"   Max cost: {traj_costs.max().item():.2f}")
        
        self._debug_counter += 1
        
        return weighted_costs


         
        
# This was getting out of scope so I will work on next version.
#TODO
# class ObstaclesCritic:
#     """
#     Distance-based obstacle cost for Robot Centric STVL costmap (0.0 to 1.0).
#     Key: Costmap origin moves with robot!

#     """
#     def __init__(
#         self,
#         weight: float = 50.0,
#         critical_threshold: float = 0.8,
#         lethal_threshold: float = 0.95,
#         critical_cost: float = 300.0,
#         collision_cost: float = 1e6,
#         stride: int = 5,
#         resolution: float = 0.1,
#         repulsion_weight: float = 2.0
#     ):
#         self.weight = weight
#         self.critical_threshold = critical_threshold
#         self.lethal_threshold = lethal_threshold
#         self.critical_cost = critical_cost
#         self.collision_cost = collision_cost
#         self.stride = stride
#         self.resolution = resolution
#         self.repulsion_weight = repulsion_weight

#     def compute(
#         self,
#         trajectories: torch.Tensor,
#         costmap: torch.Tensor,
#         grid_origin: torch.Tensor  # [2] current grid origin
#     ) -> torch.Tensor:
#         K, T, _ = trajectories.shape
#         H, W = costmap.shape
        
#         # Sample positions
#         positions = trajectories[:, ::self.stride, :2]
#         T_sampled = positions.shape[1]
        
#         # Grid coordinates (robot-centric!)
#         grid_x = ((positions[:, :, 0] - grid_origin[0]) / self.resolution).long()
#         grid_y = ((positions[:, :, 1] - grid_origin[1]) / self.resolution).long()
#         grid_x = torch.clamp(grid_x, 0, W - 1)
#         grid_y = torch.clamp(grid_y, 0, H - 1)
        
#         # Lookup costs
#         costs = costmap[grid_x, grid_y].float()  # [K, T'] ~[0, 1]
        
#         penalties = torch.zeros_like(costs)
        
#         # Collision zone (>=0.95)
#         collision_mask = costs >= self.lethal_threshold
#         penalties[collision_mask] = self.collision_cost
        
#         # Critical zone (0.8-0.95)
#         critical_mask = (costs >= self.critical_threshold) & (costs < self.lethal_threshold)
#         penalties[critical_mask] = self.critical_cost
        
#         # Repulsion zone (0.2-0.8): Exponential increase
#         repulsion_mask = (costs >= 0.2) & (costs < self.critical_threshold)
#         penalties[repulsion_mask] = self.repulsion_weight * torch.exp(
#             4.0 * costs[repulsion_mask]
#         )
        
#         # Free space (0.0-0.2): Very small cost
#         free_mask = costs < 0.2
#         penalties[free_mask] = costs[free_mask] * 5.0
        
#         # Sum over trajectory
#         total_cost = torch.sum(penalties, dim=1)  # [K]
        
#         return self.weight * total_cost


class PathAlignCritic:
    """

    Matches trajectory shape to path shape using integrated distance alignment.
    
    - For each trajectory point at distance D, finds path point also at distance D
    - Penalizes XY deviation (optionally angular) between matched points
    - Averages over all sampled trajectory points (strided for speed)
    - Early exits: near goal (<0.5m), early progress (<20 pts), blocked path (>7%)
    
    Args:
        weight: 10.0 (moderate penalty for shape deviation)
        power: 1 (linear)
        max_path_occupancy_ratio: 0.07 (disable if >7% path blocked)
        offset_from_furthest: 20 (wait for 20 points progress)
        trajectory_point_step: 4 (check every 4th point)
        threshold_to_consider: 0.5m (deactivate near goal)
        use_path_orientations: False (True = include angular error)
    """
    def __init__(
        self,
        weight: float = 10.0,
        power: int = 1,
        max_path_occupancy_ratio: float = 0.05,
        offset_from_furthest: int = 5,
        trajectory_point_step: int = 4,
        threshold_to_consider: float = 0.5,
        use_path_orientations: bool = False
    ):
        self.weight = weight
        self.power = power
        self.max_path_occupancy_ratio = max_path_occupancy_ratio
        self.offset_from_furthest = offset_from_furthest
        self.trajectory_point_step = trajectory_point_step
        self.threshold_to_consider = threshold_to_consider
        self.use_path_orientations = use_path_orientations
        self._debug_counter = 0

    def compute(self, data: 'CriticData')-> torch.Tensor:
        """
        Compute path align cost

        Args:
            data: CriticData containing all necessary information

        Returns:
            costs: [K] - Cost for each trajectory
        """
        K, T = data.trajectories.shape[:2]
        device = data.trajectories.device

        # Early Returns 
        if data.local_path_length < self.threshold_to_consider:
            return torch.zeros(K, device=device)   # Distance to goal deactivate this
        
        # Path is already pruned! data.furthest_reached_idx  is always 0 
        # We need to compute how far trajectories reached on this pruned path
        path_segments = data.path # already starts at robot

        if len(path_segments) < self.offset_from_furthest:
            return torch.zeros(K, device=device)
        
        # Compute furthest point trajectories reach
        traj_endpoints = data.trajectories[:, -1, :2]  # [K, 2]
        path_xy_full = path_segments[:, :2]  # [N, 2]

        # Distance from each trajectory endpoint to each path point
        distances = torch.cdist(traj_endpoints, path_xy_full)  # [K, N]
        closest_path_indices = torch.argmin(distances, dim=1)  # [K]
        furthest_path_point = torch.max(closest_path_indices).item()

        # Use path from robot (0) to where trajectories reach
        path_segments_count = furthest_path_point + 1
        

        if path_segments_count < self.offset_from_furthest:
            return torch.zeros(K, device=device)  # Early exit as per above
        # Calculating Path segment which is [Start: Robot position]
        # Check if this path is not occupied by anything if its good we have clean poisiton
        # which robot should take 
        if data.path_valid_flags is not None:
            valid_flags = data.path_valid_flags[:path_segments_count]
            invalid_count =(~valid_flags).sum().float()

            if path_segments_count > 0:
                occupancy_ratio = invalid_count / path_segments_count
                if (occupancy_ratio > self.max_path_occupancy_ratio and 
                    invalid_count > 2.0):
                    
                    return torch.zeros(K, device=device)
            
        
        path_xy = path_segments[:path_segments_count, :2] #[P, 2]
        path_theta = path_segments[:path_segments_count, 2] # [P]
        path_distances = data.path_integrated_distances[:path_segments_count]


        if data.path_valid_flags is not None:
            path_valid = data.path_valid_flags[:path_segments_count] #[P]
        else:
            path_valid = torch.ones(path_segments_count, dtype=torch.bool, device=device) # Placing algo
        
        if self._debug_counter % 200 == 0:
            print(f"🔍 PathAlign (FIXED - Nav2 Style):")
            print(f"   Pruned path length: {len(path_segments)}")
            print(f"   Trajectories reach: {furthest_path_point} waypoints ahead")
            print(f"   Using segment: [0:{path_segments_count}]")
            print(f"   Path distances: [0.000, {path_distances[-1].item():.3f}]")
        # Stride trajectories

        stride = self.trajectory_point_step
        T_stride = (T-1) // stride + 1

        traj_x = data.trajectories[:, ::stride, 0] #[K, T_stride]
        traj_y = data.trajectories[:, ::stride, 1] #[K, T_stride]
        traj_theta = data.trajectories[:, ::stride, 2] #[K, T_stride]

        # Compute all trajectory integrated distance at once
        dx = traj_x[:, 1:] - traj_x[:, :-1] # next - current  # [K ,T_stride-1]
        dy = traj_y[:, 1:] - traj_y[:, :-1] # next - current  # [K, T_stride-1]
        segment_lengths = torch.sqrt(dx**2 + dy**2) # [K, T_stride-1]

        # Cumulative distance with zero prepended
        zeros = torch.zeros((K,1), device=device)
        traj_distances = torch.cat([
            zeros, 
            torch.cumsum(segment_lengths, dim=1)
        ], dim=1) #[K, T_stride]

        # Flatten and batch process All trajectory points
        # Skip first point (distance = 0)
        traj_x_flat = traj_x[:, 1:].reshape(-1) #[K*(T_stride-1)]
        traj_y_flat = traj_y[:, 1:].reshape(-1) #[K*(T_stride-1)]
        traj_theta_flat = traj_theta[:,1:].reshape(-1)  #[K*(T_stride-1)]
        traj_dist_flat = traj_distances[:,1:].reshape(-1)  #[K*(T_stride-1)]

        # Total number of points to process
        N_points = len(traj_dist_flat)

        # Batch Searchsorted : Find all matching path indices at once
        path_indices = torch.searchsorted(
            path_distances.contiguous(), #[P]
            traj_dist_flat.contiguous(), #[N_points]
            right = False  
        ) # [N_points]

        # Clamp to valid range
        path_indices = torch.clamp(path_indices, 0, len(path_xy) - 1)

        # Pick closer of idx-1 or idx
        # For indices > 0, check if previous index is closer 
        prev_indices = torch.clamp(path_indices-1, min=0)  # [N_points] Take previous of respective path indices

        # Distances to current and previous indices
        curr_dist_diff  = torch.abs(
            path_distances[path_indices] - traj_dist_flat
        ) #[N_points]
        prev_dist_diff = torch.abs(
            path_distances[prev_indices] - traj_dist_flat
        )   # [N_points]

        # Use previous index where its closer
        use_prev = (prev_dist_diff < curr_dist_diff) & (path_indices > 0) #Take closer one mask
        path_indices = torch.where(use_prev, prev_indices, path_indices) # where(mask, if true this, else that)

        # Get all path points at once (advanced indexing)

        matched_path_xy = path_xy[path_indices]  # [N_points, 2]
        matched_path_theta = path_theta[path_indices] # [N_points]
        matched_path_valid = path_valid[path_indices] # [N_points]


        # Compute all XY deviations at once
        dx_all = matched_path_xy[:, 0] - traj_x_flat # [N_points]
        dy_all = matched_path_xy[:, 1] - traj_y_flat # [N_points]
        xy_dist_all = torch.sqrt(dx_all**2 + dy_all**2) #[N_points]

        if self.use_path_orientations:
            #angular error
            dyaw_all = matched_path_theta - traj_theta_flat #[N_points]
            dyaw_all = torch.atan2(
                torch.sin(dyaw_all),
                torch.cos(dyaw_all)
            ) # Normalize to [-pi to pi]
             
            # Combined distance 
            total_dist_all = torch.sqrt(
                xy_dist_all**2 + dyaw_all**2
            ) # [N_points]

        else:
            total_dist_all = xy_dist_all #[N_points]

        
        # Mask invalid points and reshape
        # zero out invalid points
        total_dist_all = torch.where(
            matched_path_valid,
            total_dist_all,
            torch.zeros_like(total_dist_all)
        ) # [N_points]  #where(mask, if true this, else that)

        # Reshape back to [K, T_stride-1]
        total_dist_per_traj = total_dist_all.reshape(K, T_stride - 1)
        valid_per_traj = matched_path_valid.reshape(K, T_stride - 1)

        # Sum and average per trajectory
        summed_dist = torch.sum(total_dist_per_traj, dim=1) #[K]

        # Count valid samples per trajectory
        num_samples = torch.sum(valid_per_traj.float(), dim=1) #[K]

        # average avoid division zero condition 
        costs = torch.where(
            num_samples > 0,
            summed_dist / num_samples,
            torch.zeros_like(summed_dist)
        ) #[K]

        # Apply weight and power
        weighted_costs = costs * self.weight
        if self.power > 1:
            weighted_costs = weighted_costs ** self.power

        if self._debug_counter % 200 == 0:
            print(f"🟣 PathAlignCritic (VECTORIZED):")
            print(f"   Path segments: {path_segments_count}")
            print(f"   Trajectories: {K}")
            print(f"   Points per trajectory: {T_stride-1}")
            print(f"   Total points processed: {N_points}")
            print(f"   Valid path points: {path_valid.sum().item()}/{len(path_valid)}")
            print(f"   Cost range: [{weighted_costs.min().item():.3f}, "
                  f"{weighted_costs.max().item():.3f}]")
            print(f"   Mean cost: {weighted_costs.mean().item():.3f}")


        if self._debug_counter % 200 == 0:
            robot_x = data.current_pose[0].item()
            robot_y = data.current_pose[1].item()
            
            path_first_x = path_xy[0, 0].item()
            path_first_y = path_xy[0, 1].item()
            
            path_last_x = path_xy[-1, 0].item()
            path_last_y = path_xy[-1, 1].item()
            
            traj_first_x = data.trajectories[0, -1, 0].item()
            traj_first_y = data.trajectories[0, -1, 1].item()
            
            print(f"🚨 PATH DIRECTION CHECK:")
            print(f"   Robot: ({robot_x:.2f}, {robot_y:.2f})")
            print(f"   Path first point: ({path_first_x:.2f}, {path_first_y:.2f})")
            print(f"   Path last point: ({path_last_x:.2f}, {path_last_y:.2f})")
            print(f"   Trajectory endpoint: ({traj_first_x:.2f}, {traj_first_y:.2f})")
            
            # Use math.sqrt for Python floats
            dist_first = math.sqrt((robot_x - path_first_x)**2 + (robot_y - path_first_y)**2)
            dist_last = math.sqrt((robot_x - path_last_x)**2 + (robot_y - path_last_y)**2)
            print(f"   Distance to path first: {dist_first:.2f}m (should be ~0, actually {dist_first:.2f})")
            print(f"   Distance to path last: {dist_last:.2f}m (should be ~0, actually {dist_last:.2f})")
            
            # THIS WILL SHOW THE BUG:
            if dist_first > 0.5:
                print(f"   ⚠️ BUG CONFIRMED: Path first point is {dist_first:.2f}m behind robot!")
            if dist_last < 0.3:
                print(f"   ⚠️ BUG CONFIRMED: Path last point is at robot, not ahead!")
            
        self._debug_counter += 1
        
        return weighted_costs



class PathAngleCritic: # done
    """
    Aligns trajectory heading with IMMEDIATE path direction.
    - Uses furthest_reached_idx + offset (MONOTONIC - never jumps backward!)
    - Uses actual path heading to normalize desired heading
    - Only activates if remaining path > threshold
    - Early exits if robot already well-aligned (< 45°)

    Args:
    - weight: 2.2 (NOT 20.0!)
    - offset_from_furthest: 4 points
    - threshold_to_consider: 0.5m (remaining path)
    - max_angle_to_furthest: 0.785398 rad (45°)
    - power: 1
    """

    def __init__(
        self,
        weight: float = 2.2,
        offset_from_furthest: int = 4,
        threshold_to_consider: float = 0.5,
        max_angle_to_furthest: float = 1.5708,   # 45 degree early exit 0.785398
        power: int = 1
    ):
        self.weight = weight
        self.offset_from_furthest = offset_from_furthest
        self.threshold_to_consider = threshold_to_consider
        self.max_angle_to_furthest = max_angle_to_furthest
        self.power = power
        self._debug_counter = 0

    def compute(
        self,
        data: 'CriticData'
    )-> torch.Tensor:
        """
        Compute path angle alignment cost.
        
        Args:
            data: CriticData containing all necessary information
        
        Returns:
            costs: [K] - Cost for each trajectory
        """
        K = data.trajectories.shape[0]
        device = data.trajectories.device
        
        # Check activation with remaining path distance (Turn off if near goal)
        if data.local_path_length < self.threshold_to_consider:
            return torch.zeros(K, device=device)
        
        # Use furthest_reached + offset(Monotonic)
        path_len = data.path.shape[0]
        target_idx = min(
            data.furthest_reached_idx + self.offset_from_furthest,
            path_len -1
        )

        # Get target point position and heading
        target_x = data.path[target_idx, 0]
        target_y = data.path[target_idx,1]
        target_heading = data.path[target_idx, 2] # From path

        # Early Exit if robot already well aligned
        robot_x = data.current_pose[0]
        robot_y = data.current_pose[1]
        robot_heading = data.current_pose[2]

        # desired heading from robot to target
        dx = target_x - robot_x
        dy = target_y - robot_y
        desired_heading = torch.atan2(dy,dx)

        # Normalize to path heading (with in +/- 90 degree)
        angle_diff = desired_heading - target_heading
        normalized_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # If > 90 degree away from path heading flip by 180 degree
        if torch.abs(normalized_diff) > 1.5708: # > pi/2
            desired_heading = torch.atan2(
                torch.sin(desired_heading + 3.14159),
                torch.cos(desired_heading + 3.14159)
            )
        
        # Check robot alignment

        robot_error = torch.atan2(
            torch.sin(desired_heading - robot_heading),
            torch.cos(desired_heading - robot_heading)
        )
        
        if torch.abs(robot_error) < self.max_angle_to_furthest:
            # Early exit
            if self._debug_counter % 200 == 0:
                print(f"🔵 PathAngleCritic: Already aligned ({torch.abs(robot_error) * 180 / 3.14159:.1f}° < 45°)")
            self._debug_counter += 1
            return torch.zeros(K, device=device)
        
        # Compute costs for trajectory endpoints
        traj_end_x = data.trajectories[: ,-1, 0] #[K]
        traj_end_y = data.trajectories[:, -1, 1] # [K]
        traj_end_heading = data.trajectories[:, -1, 2] #[K]

        # Desired heading: trajectory endpoint-> target point
        diff_x = target_x - traj_end_x #[K]
        diff_y = target_y - traj_end_y #[K]
        yaws_to_target = torch.atan2(diff_y,diff_x) #[K]

        # Normalize to path heading (within +/- 90 degree)
        angle_diff = yaws_to_target - target_heading #[K]
        normalized_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)) # [K]

        # flip heading that are > 90 degree away from path heading
        flip_mask = torch.abs(normalized_diff) > 1.5708  # [K] boolean
        yaws_corrected = yaws_to_target.clone()
        yaws_corrected[flip_mask] = torch.atan2(
            torch.sin(yaws_to_target[flip_mask] + 3.14159),
            torch.cos(yaws_to_target[flip_mask] + 3.14159)
        )

        # Angular error betweeen trajectory heading and corrected desired heading
        angle_error = torch.atan2(
            torch.sin(traj_end_heading - yaws_corrected),
            torch.cos(traj_end_heading - yaws_corrected)
        )

        # Compute cost
        cost = torch.abs(angle_error) * self.weight

        if self.power > 1:
            cost = cost ** self.power

        # Debug
        if self._debug_counter % 200 == 0:
            print(f"🔵 PathAngleCritic ACTIVE")
            print(f"   Target: idx={target_idx} (furthest={data.furthest_reached_idx} + {self.offset_from_furthest})")
            print(f"   Target pos: ({target_x:.2f}, {target_y:.2f}), heading={target_heading * 180 / 3.14159:.1f}°")
            print(f"   Robot error: {torch.abs(robot_error) * 180 / 3.14159:.1f}° (threshold=45°)")
            print(f"   Avg traj error: {torch.abs(angle_error).mean().item() * 180 / 3.14159:.1f}°")
        
        self._debug_counter += 1

        return cost
    

class PathFollowCritic: #done
    """
    Drives Trajectory endpoint toward specific path point ahead of progress.
    
    - Targets SINGLE point (not a section!)
    - Uses furthest_reached_idx + offset (monotonic progress)
    - Skips blocked path points (path validity checking)
    - Only activates if remaining path > threshold
    args:
    - weight: 5.0
    - offset_from_furthest: 6 points
    - threshold_to_consider: 1.4m (remaining path)
    - power: 1
    """
    def __init__(
        self,
        weight: float = 5.0,              
        offset_from_furthest: int = 6,    
        threshold_to_consider: float = 1.4, 
        power: int = 1                    
    ):
        self.weight = weight
        self.offset_from_furthest = offset_from_furthest
        self.threshold_to_consider = threshold_to_consider
        self.power = power
        self._debug_counter = 0

    def compute(
        self,
        data: 'CriticData'
    ) -> torch.Tensor:
        """
        Compute path following cost.
        
        Args:
            data: CriticData containing all necessary information
        
        Returns:
            costs: [K] - Cost for each trajectory
        """
        K = data.trajectories.shape[0]
        device = data.trajectories.device

        if data.path.shape[0] < 2 or data.local_path_length < self.threshold_to_consider:
            return torch.zeros(K, device=device)
        path_size = data.path.shape[0]-1
        target_idx = min(
            data.furthest_reached_idx + self.offset_from_furthest,
            path_size
        )
        valid = False
        while not valid and target_idx < path_size - 1:
            valid = data.path_valid_flags[target_idx].item()
            if not valid:
                target_idx +=1
                if self._debug_counter % 200 == 0:
                    print(f"   ⚠️ Skipping blocked point at idx {target_idx-1}")

        # Get target point (single point)
        target_xy = data.path[target_idx, :2] #[2]

        traj_position = data.trajectories[:, -1, :2] #[K, 2]

        diff = traj_position - target_xy #[K, 2]
        
        distance = torch.linalg.norm(diff, ord=2, dim=-1) #[K]

        cost = distance * self.weight

        if self.power >1:
            cost = cost ** self.power

        # Debug
        if self._debug_counter % 200 == 0:
            print(f"🟢 PathFollowCritic ACTIVE")
            print(f"   Target: idx={target_idx} (furthest={data.furthest_reached_idx} + {self.offset_from_furthest})")
            print(f"   Target pos: ({target_xy[0]:.2f}, {target_xy[1]:.2f})")  # ✅ Fixed!
            print(f"   Valid: {valid}")
            print(f"   Avg endpoint distance: {distance.mean().item():.3f}m")  # ✅ Fixed!
            print(f"   Remaining path: {data.local_path_length:.2f}m")

        self._debug_counter += 1
        
        return cost
    




class GoalCritic: # Done
    """
    Attracts robot to final goal position.
    - Activates when remaining path < threshold 
    - Averages distance over ALL trajectory points (not just endpoint)
    - Supports power parameter for non-linear penalty

    - weight: 5.0
    - threshold_to_consider: 1.4m (remaining path distance)
    - power: 1 (linear penalty)
    """
    def __init__(
        self,
        weight: float = 5.0,
        threshold_to_consider: float = 1.4,
        power: int = 1   
    ):
        self.weight = weight
        self.threshold_to_consider = threshold_to_consider
        self.power = power
        self._debug_counter = 0

    def compute(
        self,
        data: 'CriticData'
    )-> torch.Tensor:
        """
        Compute goal attraction cost.
        Args:
            data: CriticData containing all necessary information
        Returns:
            costs: [K] - Cost for each trajectory
        """
        K = data.trajectories.shape[0]
        device = data.trajectories.device

        # Use remaining path distance,
        if data.local_path_length > self.threshold_to_consider:
            return torch.zeros(K, device=device)
        
        if self._debug_counter % 200 == 0:
            print(f"🎯 GoalCritic ACTIVE!")
            print(f"   Remaining path: {data.local_path_length:.2f}m < {self.threshold_to_consider:.2f}m")
            current_dist = torch.norm(data.current_pose[:2] - data.goal).item()
            print(f"   Straight-line dist: {current_dist:.2f}m (for reference)")
        
        self._debug_counter += 1

        # Extract goal position (XY only)
        goal_xy = data.goal #[2]

        # Distance from all trajectory points to goal
        traj_position = data.trajectories[:,:,:2] # [K, T, 2]

        diff = traj_position - goal_xy # [K, T ,2]

        # Distance from all points: [K, T]
        distances = torch.linalg.norm(diff, ord=2, dim=-1) #[K,T]

        avg_distance = torch.mean(distances, dim=1) #[K] # we need avg along all time step so how overall tis traj scored

        cost = avg_distance * self.weight

        if self.power >1:
            cost = cost ** self.power
        return cost
    


    
class GoalAngleCritic: #done
    """
    Aligns final heading with goal direction.
    - Activates when remaining path < threshold (0.5m default)
    - Uses goal heading from path (not computed from XY!)
    - Averages angular error over ALL trajectory points
    - Supports power parameter

    Args:
    - weight: 3.0
    - threshold_to_consider: 0.5m (remaining path distance)
    - power: 1

    """
    def __init__(
        self,
        weight: float = 3.0,
        threshold_to_consider: float = 0.5,
        power: int = 1
        ):
        self.weight = weight
        self.threshold_to_consider = threshold_to_consider
        self.power = power
        self._debug_counter = 0
    
    def compute(
        self,
        data: 'CriticData'
    )-> torch.Tensor:
        """
        Compute goal heading alignment cost.
        
        Args:
            data: CriticData containing all necessary information
        
        Returns:
            costs: [K] - Cost for each trajectory
        """

        K = data.trajectories.shape[0]
        device = data.trajectories.device

        if data.local_path_length > self.threshold_to_consider:
            return torch.zeros(K, device=device)
        
        # Debug output
        if self._debug_counter % 200 == 0:
            print(f"📐 GoalAngleCritic ACTIVE!")
            print(f"   Remaining path: {data.local_path_length:.2f}m < {self.threshold_to_consider:.2f}m")
            print(f"   Goal heading: {data.goal_heading:.2f} rad ({data.goal_heading * 180 / 3.14159:.1f}°)")
        
        self._debug_counter += 1
        goal_heading = data.goal_heading # Already in critic data

        trajectory_heading = data.trajectories[:, :, 2] # [K, T]

        # Compute angular distance for all points
        # shortest_angular_distance(traj_heading, goal_heading)
        angle_diff = trajectory_heading - goal_heading  #[K,T]

        # Wrap to [-pi,pi]
        angle_error = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)) # [K,T]
        
        angle_error_abs = torch.abs(angle_error) # [K,T]

        #Average over time dimension
        avg_angle_error = torch.mean(angle_error_abs, dim = 1) #[K]

        # Apply weight
        cost = avg_angle_error * self.weight

        # Apply power if >1
        if self.power >1:
            cost = cost ** self.power
        
        return cost
    
    
class ConstraintCritic: #done
    """
    Enforces Velocity Limits(soft constraints)
    """
    def __init__(
        self,
        weight: float = 4.0,
        power: int = 1     
        ):
        """
        Args:
            weight: Penalty weight for constraint violations
            power: Exponent for cost (1 = linear, 2 = quadratic)
        
        Note: v_max, v_min, w_max, dt now come from CriticData!
        """
        self.weight = weight
        self.power = power
        self._debug_counter = 0

    def compute(
        self,
        data: 'CriticData'
    )-> torch.Tensor:
        """
        Compute constraint violation penalties.
        
        Args:
            data: Unified critic data structure
        
        Returns:
            costs: [K] Constraint violation cost for each trajectory
        """
        K, T = data.v_samples.shape

        # Get limits from CriticData
        v_max = data.v_max
        v_min = data.v_min
        w_max = data.w_max
        dt = data.dt

        # Linear velocity violations
        v_above_max = torch.clamp(data.v_samples - v_max, min=0.0)
        v_below_min = torch.clamp(v_min - data.v_samples, min=0.0)
        v_violations = v_above_max + v_below_min  #[K,T]

        # Angular velocity violations
        w_violations = torch.clamp(torch.abs(data.w_samples) - w_max, min=0.0)  # [K,T]
        
        # Combine and multiply by dt
        total_violations = (v_violations + w_violations) * dt #[K,T]

        # Sum over trajectory
        summed_violations = torch.sum(total_violations, dim=1) #[K]

        weighted_cost = summed_violations * self.weight
        if self.power > 1:
            weighted_cost = weighted_cost ** self.power
        
        # Debug
        if self._debug_counter % 200 == 0:
            v_viol_count = (v_violations > 0).any(dim=1).sum().item()
            w_viol_count = (w_violations > 0).any(dim=1).sum().item()
            print(f"⚫ ConstraintCritic:")
            print(f"   V violations: {v_viol_count}/{K} trajectories")
            print(f"   W violations: {w_viol_count}/{K} trajectories")
            print(f"   Cost range: [{weighted_cost.min().item():.2f}, {weighted_cost.max().item():.2f}]")
        
        self._debug_counter += 1
        
        return weighted_cost


    
class TwirlingCritic: #done
    """
    Reduces excessive rotation when far from goal.
    - Penalizes rotation (mean of |w| over time)
    - Deactivates NEAR goal to allow final orientation adjustment
    - Uses remaining path distance (not straight-line)

    Args:
    - weight: 10.0
    - power: 1
    - threshold: ~0.5m (approximate goal tolerance from Nav2's goal_checker)

    """
    def __init__(
        self,
        weight: float = 10.0,
        power: int= 1,
        threshold_to_consider: float = 0.5
    ):
        self.weight = weight
        self.power = power
        self.threshold_to_consider = threshold_to_consider
        self._debug_counter = 0

    def compute(
        self,
        data: 'CriticData'
    )-> torch.Tensor:
        """
        Compute twirling penalty cost.
        
        Args:
            data: CriticData containing all necessary information
        
        Returns:
            costs: [K] - Cost for each trajectory
        """
        K = data.w_samples.shape[0]
        device = data.w_samples.device

        if data.local_path_length < self.threshold_to_consider:
            if self._debug_counter % 200 == 0:
                print(f"🔄 TwirlingCritic INACTIVE (near goal)")
                print(f"   Remaining path: {data.local_path_length:.2f}m < {self.threshold_to_consider:.2f}m")
            self._debug_counter += 1
            return torch.zeros(K, device=device)
        

        # mean of absolute angular velocities (not sum!)
        avg_rotation = torch.mean(torch.abs(data.w_samples), dim=1)   #[K]

        # Debug output
        if self._debug_counter % 200 == 0:
            print(f"🔄 TwirlingCritic ACTIVE")
            print(f"   Remaining path: {data.local_path_length:.2f}m")
            print(f"   Avg rotation penalty: {avg_rotation.mean().item():.3f} rad/s")
        
        self._debug_counter += 1

        cost = avg_rotation * self.weight

        if self.power > 1:
            cost = cost ** self.power

        return cost
    



class PreferForwardCritic: #done
    """
    Discourages backward motion.
    - weight: 5.0
    - threshold_to_consider: 0.5m (path length, NOT velocity!)
    - power: 1
    """
    def __init__(
        self,
        weight: float = 5.0,
        threshold_to_consider: float = 0.5, 
        power: int = 1
    ):
        """
        Args:
            weight: Penalty weight for backward motion
            threshold_to_consider: Only penalize if remaining path > this distance (meters)
                                  This allows backing up near goal!
            power: Exponent for cost (1 = linear, 2 = quadratic)
        """
        self.weight = weight
        self.threshold_to_consider = threshold_to_consider
        self.power = power
        self._debug_counter = 0

    def compute(self, data: 'CriticData') -> torch.Tensor:
        """
        Penalize backward motion when far from goal.
        
        Nav2 formula:
        cost = (sum_over_time(max(-vx, 0) * dt) * weight) ^ power
        
        Args:
            data: Unified critic data structure
        
        Returns:
            costs: [K] Backward motion penalty for each trajectory
        """

        K, T = data.v_samples.shape
        device = data.v_samples.device

        #  Early return if close to goal
        # This allows backing into parking spots or final positioning
        if data.local_path_length < self.threshold_to_consider:
            return torch.zeros(K, device=device)

        # Penalize only negative velocities (backward motion)
        # max(-vx, 0) means:
        #   vx = -0.2 (backward) → max(0.2, 0) = 0.2 penalty
        #   vx = +0.3 (forward)  → max(-0.3, 0) = 0 (no penalty)
        
        backward_penalties = torch.clamp(-data.v_samples, min=0.0)  # [K, T]

        # Multiply by dt (convert velocity to distance)
        backward_penalties = backward_penalties * data.dt  # [K, T]

        # Sum over trajectory
        total_penalty = torch.sum(backward_penalties, dim=1)  # [K]

        # Apply weight and power
        weighted_cost = total_penalty * self.weight  # [K]
        if self.power > 1:
            weighted_cost = weighted_cost ** self.power
        
        # Debug
        if self._debug_counter % 200 == 0:
            backward_count = (data.v_samples < 0).any(dim=1).sum().item()
            print(f"⚫ PreferForwardCritic:")
            print(f"   Local path length: {data.local_path_length:.2f}m")
            print(f"   Threshold: {self.threshold_to_consider:.2f}m")
            print(f"   Active: {data.local_path_length >= self.threshold_to_consider}")
            print(f"   Backward trajectories: {backward_count}/{K}")
            if weighted_cost.max() > 0:
                print(f"   Max penalty: {weighted_cost.max().item():.2f}")
        
        self._debug_counter += 1
        
        return weighted_cost

    
class DeadbandCritic: #done
    """
    Penalizes velocities below deadband thresholds.
    
    Encourages robot to move decisively rather than creeping slowly.
    - weight: 35.0 (high to strongly discourage creeping)
    - v_deadband: 0.0 (typically disabled for vx)
    - vy_deadband: 0.0 (for holonomic robots)
    - w_deadband: 0.0 (typically disabled for wz)
    - power: 1
    """
    def __init__(
        self,
        weight: float = 35.0,      
        v_deadband: float = 0.1,   
        w_deadband: float = 0.05,   
        power: int = 1
    ):
        """
        Args:
            weight: Penalty weight (high to strongly discourage slow motion)
            v_deadband: Linear velocity deadband (m/s)
            w_deadband: Angular velocity deadband (rad/s)
            power: Exponent for cost (1 = linear, 2 = quadratic)
        
        Note: Typically all deadzones are 0.0 (disabled) unless your
              hardware has specific deadband requirements.
        """
        self.weight = weight
        self.v_deadband = abs(v_deadband)  # Nav2 uses fabs()
        self.w_deadband = abs(w_deadband)
        self.power = power
        self._debug_counter = 0
    
    def compute(self, data: 'CriticData') -> torch.Tensor:
        """
        Penalize velocities below deadband thresholds.
        
        Nav2 formula (DiffDrive):
        cost = (sum_over_time(
                    max(v_deadband - |vx|, 0) + 
                    max(w_deadband - |wz|, 0)
                ) * dt * weight) ^ power
        
        Args:
            data: Unified critic data structure
        
        Returns:
            costs: [K] Deadband violation cost for each trajectory
        """
        K, T = data.v_samples.shape
        device = data.v_samples.device
        
        # ✅ Early return if deadzones disabled (Nav2 optimization)
        if self.v_deadband == 0.0 and self.w_deadband == 0.0:
            return torch.zeros(K, device=device)
        
        # Nav2: (deadband - |velocity|).max(0)
        # Penalizes by HOW MUCH velocity is below deadband
        
        # Linear velocity penalty
        v_abs = torch.abs(data.v_samples)  # [K, T]
        v_penalty = torch.clamp(self.v_deadband - v_abs, min=0.0)  # [K, T]
        
        # Angular velocity penalty
        w_abs = torch.abs(data.w_samples)  # [K, T]
        w_penalty = torch.clamp(self.w_deadband - w_abs, min=0.0)  # [K, T]
        
        # Combine penalties
        total_penalty = v_penalty + w_penalty  # [K, T]

        # Multiply by dt (convert to distance units)
        total_penalty = total_penalty * data.dt  # [K, T]


        # Sum over trajectory
        summed_penalty = torch.sum(total_penalty, dim=1)  # [K]
        
        # Apply weight and power
        weighted_cost = summed_penalty * self.weight  # [K]
        if self.power > 1:
            weighted_cost = weighted_cost ** self.power
        
        # Debug
        if self._debug_counter % 200 == 0:
            v_in_deadband = (v_abs < self.v_deadband).any(dim=1).sum().item()
            w_in_deadband = (w_abs < self.w_deadband).any(dim=1).sum().item()
            print(f"⚫ DeadbandCritic:")
            print(f"   V deadband: {self.v_deadband} m/s")
            print(f"   W deadband: {self.w_deadband} rad/s")
            print(f"   V violations: {v_in_deadband}/{K}")
            print(f"   W violations: {w_in_deadband}/{K}")
            if weighted_cost.max() > 0:
                print(f"   Max penalty: {weighted_cost.max().item():.2f}")
        
        self._debug_counter += 1
        
        return weighted_cost
    


        
