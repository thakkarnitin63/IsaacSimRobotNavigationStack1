# navigation_stack/planner/global_planner.py

import sys
import os
import yaml
import numpy as np
import math
from contextlib import ExitStack
from PIL import Image, ImageDraw
import importlib_resources as resources
from scipy.ndimage import binary_dilation
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

try:
    from ccma import CCMA
    CCMA_AVAILABLE = True
except ImportError:
    CCMA_AVAILABLE = False
    print("WARNING: CCMA not available. Path smoothing disabled.")


class GlobalPlanner:
    """
    A* global planner with optional CCMA smoothing and heading calculation.
    Returns paths in format [N, 3] with [x, y, heading_rad].
    """
    
    def __init__(self, 
                 use_ccma_smoothing: bool = True,
                 ccma_w_ma: int = 5,
                 ccma_w_cc: int = 3,
                 debug: bool = False):
        """
        Args:
            use_ccma_smoothing: Enable corner smoothing
            ccma_w_ma: Moving average window (higher = smoother corners)
            ccma_w_cc: Curvature constraint (higher = smoother curves)
            debug: Print detailed path processing info
        """
        try:
            with ExitStack() as stack:
                maps_pkg = resources.files("navigation_stack.maps")
                yaml_file = stack.enter_context(resources.as_file(maps_pkg.joinpath("map.yaml")))
                img_file = stack.enter_context(resources.as_file(maps_pkg.joinpath("map.png")))

                with open(yaml_file, 'r') as f:
                    self.map_info = yaml.safe_load(f)
                    
                self.resolution = self.map_info['resolution']
                self.origin = self.map_info['origin'] 
                self.occupied_thresh = self.map_info['occupied_thresh'] * 255
                self._load_and_process_map(img_file)
        except (ModuleNotFoundError, FileNotFoundError):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            maps_dir = os.path.join(os.path.dirname(script_dir), "maps")
            
            yaml_path = os.path.join(maps_dir, "map.yaml")
            img_path = os.path.join(maps_dir, "map.png")
            
            if not os.path.exists(yaml_path) or not os.path.exists(img_path):
                raise FileNotFoundError(f"Map files not found in {maps_dir}")
            
            with open(yaml_path, 'r') as f:
                self.map_info = yaml.safe_load(f)
            
            self.resolution = self.map_info['resolution']
            self.origin = self.map_info['origin'] 
            self.occupied_thresh = self.map_info['occupied_thresh'] * 255
            self._load_and_process_map(img_path)

        self.debug = debug
        self.use_ccma_smoothing = use_ccma_smoothing
        self.ccma = None
        
        if self.use_ccma_smoothing and CCMA_AVAILABLE:
            try:
                self.ccma = CCMA(w_ma=ccma_w_ma, w_cc=ccma_w_cc)
                if self.debug:
                    print(f"CCMA initialized (w_ma={ccma_w_ma}, w_cc={ccma_w_cc})")
            except Exception as e:
                print(f"WARNING: CCMA init failed: {e}")
                self.use_ccma_smoothing = False

    def _load_and_process_map(self, image_path):
        img = Image.open(image_path).convert('L')
        self.map_grid_raw = np.array(img)
        self.height, self.width = self.map_grid_raw.shape

        occupied_mask = self.map_grid_raw < self.occupied_thresh
        inflation_pixels = 5
        structure = np.ones((2 * inflation_pixels + 1, 2 * inflation_pixels + 1))
        inflated_mask = binary_dilation(occupied_mask, structure=structure)
        self.cost_matrix = np.where(inflated_mask, 0, 1).astype(np.int8)
        self.path_grid = Grid(matrix=self.cost_matrix)

    # Add to GlobalPlanner or run separately
    def visualize_map(self, output_path="map_debug.png"):
        """Save the cost map to see obstacles."""
        from PIL import Image
        
        # cost_matrix: 1 = free, 0 = obstacle
        vis = (self.cost_matrix * 255).astype(np.uint8)
        img = Image.fromarray(vis)
        img.save(output_path)
        print(f"Saved cost map to {output_path}")

    def _apply_ccma_smoothing(self, waypoints_xy: np.ndarray) -> np.ndarray:
        """Apply CCMA smoothing to round corners."""
        if not self.use_ccma_smoothing or self.ccma is None:
            return waypoints_xy
            
        if waypoints_xy.shape[0] < 10:
            if self.debug:
                print(f"Path too short ({len(waypoints_xy)} pts), skipping CCMA")
            return waypoints_xy

        try:
            smoothed = self.ccma.filter(waypoints_xy.astype(np.float64))
            
            if self.debug:
                print(f"CCMA: {len(waypoints_xy)} pts -> {len(smoothed)} pts")
            
            return smoothed

        except Exception as e:
            print(f"WARNING: CCMA error: {e}")
            return waypoints_xy

    def _add_heading_to_path(self, waypoints_xy: np.ndarray) -> np.ndarray:
        """Add heading column using averaged directions at corners."""
        num_points = waypoints_xy.shape[0]
        if num_points == 0:
            return np.zeros((0, 3))
        
        path_with_heading = np.zeros((num_points, 3))
        path_with_heading[:, :2] = waypoints_xy
        
        if num_points == 1:
            return path_with_heading

        for i in range(num_points):
            if i == 0:
                dx = waypoints_xy[1, 0] - waypoints_xy[0, 0]
                dy = waypoints_xy[1, 1] - waypoints_xy[0, 1]
            elif i == num_points - 1:
                dx = waypoints_xy[i, 0] - waypoints_xy[i-1, 0]
                dy = waypoints_xy[i, 1] - waypoints_xy[i-1, 1]
            else:
                dx_in = waypoints_xy[i, 0] - waypoints_xy[i-1, 0]
                dy_in = waypoints_xy[i, 1] - waypoints_xy[i-1, 1]
                dx_out = waypoints_xy[i+1, 0] - waypoints_xy[i, 0]
                dy_out = waypoints_xy[i+1, 1] - waypoints_xy[i, 1]
                
                dx = (dx_in + dx_out) / 2.0
                dy = (dy_in + dy_out) / 2.0
                
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    dx = dx_out if abs(dx_out) > 1e-6 else 1.0
                    dy = dy_out if abs(dy_out) > 1e-6 else 0.0
            
            path_with_heading[i, 2] = math.atan2(dy, dx)

        if self.debug:
            print(f"Heading: {math.degrees(path_with_heading[0,2]):.1f}° -> "
                  f"{math.degrees(path_with_heading[-1,2]):.1f}°")
        
        return path_with_heading

    def world_to_grid(self, world_x, world_y):
        grid_x = int((world_x - self.origin[0]) / self.resolution)
        grid_y_from_origin = int((world_y - self.origin[1]) / self.resolution)
        grid_y = self.height - 1 - grid_y_from_origin 
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        world_x = (grid_x * self.resolution) + self.origin[0]
        grid_y_from_origin = self.height - 1 - grid_y
        world_y = (grid_y_from_origin * self.resolution) + self.origin[1]
        return world_x, world_y

    def is_valid_grid_pos(self, grid_x, grid_y):
        if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            return False
        return self.cost_matrix[grid_y, grid_x] == 1 

    def plan_path(self, start_world, end_world):
        """
        Plan path with heading using 8-neighbor A* search.
        
        Returns:
            path_with_heading: [N, 3] array with [x, y, theta_rad]
            grid_path: A* grid nodes (for visualization)
        """
        start_grid = self.world_to_grid(start_world[0], start_world[1])
        end_grid = self.world_to_grid(end_world[0], end_world[1])
        
        if not self.is_valid_grid_pos(*start_grid):
            print(f"ERROR: Invalid start position: {start_grid}")
            return None, None
        if not self.is_valid_grid_pos(*end_grid):
            print(f"ERROR: Invalid goal position: {end_grid}")
            return None, None
        
        start_node = self.path_grid.node(*start_grid)
        end_node = self.path_grid.node(*end_grid)
        
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        grid_path, runs = finder.find_path(start_node, end_node, self.path_grid)
        self.path_grid.cleanup()
        
        if not grid_path:
            print("ERROR: No path found")
            return None, None
        
        world_waypoints = np.array([self.grid_to_world(n.x, n.y) for n in grid_path])
        smoothed = self._apply_ccma_smoothing(world_waypoints)
        path_with_heading = self._add_heading_to_path(smoothed)

        # self.visualize_map()
        
        if self.debug:
            print(f"Path generated: {len(path_with_heading)} waypoints [x, y, theta]")
        
        return path_with_heading, grid_path

    def save_path_image_with_heading(self, path_with_heading: np.ndarray, 
                                     output_path: str, arrow_spacing: int = 10):
        """Visualize path with heading arrows."""
        img = Image.fromarray(self.map_grid_raw).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        grid_points = [self.world_to_grid(p[0], p[1]) for p in path_with_heading]
        if len(grid_points) > 1:
            draw.line(grid_points, fill=(255, 100, 100), width=2)

        arrow_len_world = 0.3
        for i in range(0, len(path_with_heading), arrow_spacing):
            wx, wy, theta = path_with_heading[i]
            gx, gy = self.world_to_grid(wx, wy)
            
            tip_wx = wx + arrow_len_world * math.cos(theta)
            tip_wy = wy + arrow_len_world * math.sin(theta)
            tip_gx, tip_gy = self.world_to_grid(tip_wx, tip_wy)
            
            draw.line([(gx, gy), (tip_gx, tip_gy)], fill=(0, 0, 255), width=2)
            draw.ellipse([gx-2, gy-2, gx+2, gy+2], fill=(0, 255, 0))

        if len(grid_points) > 0:
            sx, sy = grid_points[0]
            ex, ey = grid_points[-1]
            draw.rectangle([sx-3, sy-3, sx+3, sy+3], fill=(0, 255, 0))
            draw.rectangle([ex-3, ey-3, ex+3, ey+3], fill=(255, 0, 0))
        
        img.save(output_path)
        if self.debug:
            print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    print("=== GlobalPlanner Test Suite ===\n")
    
    planner = GlobalPlanner(use_ccma_smoothing=True, debug=True)
    
    # Test 1: Straight path
    print("\n--- Test 1: Straight Path ---")
    path, _ = planner.plan_path((0.0, 0.0), (5.0, 0.0))
    if path is not None:
        print(f"First heading: {math.degrees(path[0, 2]):.1f}°")
        print(f"Last heading: {math.degrees(path[-1, 2]):.1f}°")
    
    # Test 2: Diagonal path
    print("\n--- Test 2: Diagonal Path ---")
    path, _ = planner.plan_path((1.0, 2.0), (5.0, 5.0))
    if path is not None:
        corner_idx = len(path) // 2
        print("Mid-path headings:")
        for i in range(max(0, corner_idx-2), min(len(path), corner_idx+2)):
            x, y, th = path[i]
            print(f"  [{i}]: ({x:.2f}, {y:.2f}) theta={math.degrees(th):.1f}°")
    
    # Test 3: L-Shape corner smoothing
    print("\n--- Test 3: L-Shape Corner Smoothing ---")
    
    north_segment = np.array([[1.0, y] for y in np.linspace(2.0, 5.0, 8)])
    east_segment = np.array([[x, 5.0] for x in np.linspace(1.0, 5.0, 9)])
    forced_waypoints_xy = np.vstack([north_segment, east_segment[1:]])
    
    # Without CCMA
    planner_no_ccma = GlobalPlanner(use_ccma_smoothing=False, debug=False)
    path_no_ccma = planner_no_ccma._add_heading_to_path(forced_waypoints_xy)
    
    print("\nWithout CCMA:")
    corner_idx = 7
    for i in range(corner_idx-1, min(corner_idx+3, len(path_no_ccma))):
        x, y, th = path_no_ccma[i]
        print(f"  [{i}]: ({x:.2f}, {y:.2f}) theta={math.degrees(th):.1f}°")
    
    # With CCMA
    smoothed_xy = planner._apply_ccma_smoothing(forced_waypoints_xy)
    path_with_ccma = planner._add_heading_to_path(smoothed_xy)
    
    print("\nWith CCMA:")
    for i in range(len(path_with_ccma)):
        x, y, th = path_with_ccma[i]
        if 0.9 <= x <= 1.2 and 4.8 <= y <= 5.2:
            print(f"  [{i}]: ({x:.2f}, {y:.2f}) theta={math.degrees(th):.1f}°")
    
    # Analyze smoothness
    heading_changes_no_ccma = np.abs(np.diff(path_no_ccma[:, 2]))
    heading_changes_with_ccma = np.abs(np.diff(path_with_ccma[:, 2]))
    
    heading_changes_no_ccma = np.minimum(heading_changes_no_ccma, 
                                         2*np.pi - heading_changes_no_ccma)
    heading_changes_with_ccma = np.minimum(heading_changes_with_ccma, 
                                           2*np.pi - heading_changes_with_ccma)
    
    print(f"\nHeading Smoothness Analysis:")
    print(f"  Without CCMA - Max change: {math.degrees(heading_changes_no_ccma.max()):.1f}°")
    print(f"  With CCMA    - Max change: {math.degrees(heading_changes_with_ccma.max()):.1f}°")
    print(f"  Improvement: {math.degrees(heading_changes_no_ccma.max() - heading_changes_with_ccma.max()):.1f}°")
    
    # Save visualization
    output_dir = os.path.join(os.path.dirname(__file__), "..", "maps")
    output_path = os.path.join(output_dir, "test_forced_L_with_ccma.png")
    planner.save_path_image_with_heading(path_with_ccma, output_path, arrow_spacing=3)
    
    print("\nTest suite complete.")