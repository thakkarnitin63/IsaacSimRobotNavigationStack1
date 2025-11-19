# In: navigation_stack/planner/global_planner.py

import sys
import os
import yaml
import numpy as np
from contextlib import ExitStack
from PIL import Image, ImageDraw
import importlib_resources as resources
from scipy.ndimage import binary_dilation
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement


class GlobalPlanner:
    """
    Loads a map as a package resource, provides a fast A* pathfinding interface,
    and can visualize the resulting path on the map.
    """
    def __init__(self):
        """
        Initializes the planner by loading and processing map resources.
        """
        with ExitStack() as stack:
            try:
                maps_pkg = resources.files("navigation_stack.maps")
            except ModuleNotFoundError:
                print("Error: Could not find the 'navigation_stack.maps' package.")
                sys.exit(1)
            
            yaml_file = stack.enter_context(resources.as_file(maps_pkg.joinpath("map.yaml")))
            img_file = stack.enter_context(resources.as_file(maps_pkg.joinpath("map.png")))

            with open(yaml_file, 'r') as f:
                self.map_info = yaml.safe_load(f)

            # We must load the real origin from the YAML file
            self.resolution = self.map_info['resolution']
            self.origin = self.map_info['origin'] 
            self.occupied_thresh = self.map_info['occupied_thresh'] * 255

            # Load the map first so we know its size
            self._load_and_process_map(img_file)

            print(f"Map loaded. Resolution: {self.resolution}, Origin: {self.origin}")


    def _load_and_process_map(self, image_path):
        """ Loads the map image, inflates obstacles, and creates the A* grid. """
        img = Image.open(image_path).convert('L')
        map_grid_raw = np.array(img)
        
        self.map_grid_raw = map_grid_raw.copy()
        
        self.height, self.width = map_grid_raw.shape
        print(f"Map image dimensions: {self.width} (w) x {self.height} (h)")

        occupied_mask = map_grid_raw < self.occupied_thresh
        # Inflate the obstacles
        # This creates a "safety margin" around walls
        inflation_pixels = 5
        structure = np.ones((2 * inflation_pixels + 1, 2 * inflation_pixels + 1))
        inflated_mask = binary_dilation(occupied_mask, structure=structure)
        
        # Create the final cost matrix
        # 0 = impassable (obstacle), 1 = passable (free space)
        self.cost_matrix = np.where(inflated_mask, 0, 1).astype(np.int8)
        
        self.path_grid = Grid(matrix=self.cost_matrix)

        print("Cost grid created and A* finder initialized.")

    def _simplify_path(self, path, step_size=20):
        """
        Simplifies a grid path by subsampling, taking every Nth point.
        This is far more robust for MPPI than just taking corners.
        
        :param path: A list of GridNode objects from the pathfinder.
        :param step_size: Take one waypoint every N steps.
        :return: A simplified list of GridNode objects.
        """
        if not path:
            return []
            
        # --- THIS IS THE NEW, ROBUST LOGIC ---
        # Subsample the path, taking every 'step_size' point
        simplified_path = path[::step_size]
        
        # CRITICAL: Always ensure the final goal is included,
        # even if the path length isn't a multiple of step_size.
        if path[-1] not in simplified_path:
            simplified_path.append(path[-1])
        # --- END OF NEW LOGIC ---
        
        print(f"Path simplified from {len(path)} to {len(simplified_path)} waypoints (taking every {step_size}th point).")
        return simplified_path

    def world_to_grid(self, world_x, world_y):
        """Converts world coordinates to (col, row) grid coordinates."""
        # Standard conversion: just subtract origin and divide by resolution
        grid_x = int((world_x - self.origin[0]) / self.resolution)
        
        # Y-axis: image is flipped vertically (0,0 at top-left)
        # In world: Y increases upward
        # In image: row increases downward
        grid_y_from_origin = int((world_y - self.origin[1]) / self.resolution)
        grid_y = self.height - 1 - grid_y_from_origin  # Only flip Y
        
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Converts (col, row) grid coordinates to world coordinates."""
        # Reverse the conversion
        world_x = (grid_x * self.resolution) + self.origin[0]
        
        # Un-flip Y
        grid_y_from_origin = self.height - 1 - grid_y
        world_y = (grid_y_from_origin * self.resolution) + self.origin[1]
        
        return world_x, world_y

    def is_valid_grid_pos(self, grid_x, grid_y):
        """ Checks if a grid position is valid (in bounds and walkable). """
        if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            return False
        return self.cost_matrix[grid_y, grid_x] == 1 

    def plan_path(self, start_world, end_world):
        """
        Public method to plan a path from world coordinates.
        Returns: A tuple of (world_path, grid_path) or (None, None) if no path is found.
        """
        start_grid = self.world_to_grid(start_world[0], start_world[1])
        end_grid = self.world_to_grid(end_world[0], end_world[1])
        
        if not self.is_valid_grid_pos(start_grid[0], start_grid[1]):
            print(f"Error: Start position {start_grid} is on an obstacle or out of bounds.")
            return None, None
        if not self.is_valid_grid_pos(end_grid[0], end_grid[1]):
            print(f"Error: End position {end_grid} is on an obstacle or out of bounds.")
            return None, None
            
        print(f"Planning from grid {start_grid} to {end_grid}...")
        
        start_node = self.path_grid.node(start_grid[0], start_grid[1])
        end_node = self.path_grid.node(end_grid[0], end_grid[1])
        
        finder = AStarFinder()
        grid_path, runs = finder.find_path(start_node, end_node, self.path_grid)
        
        self.path_grid.cleanup()
        
        if not grid_path:
            print("No path found.")
            return None, None
            
        print(f"Path found with {len(grid_path)} waypoints (A* runs: {runs}).")
        
        # simplified_grid_path = self._simplify_path(grid_path)
        
        # 3. Convert the SIMPLIFIED path to world coordinates
        world_path = [self.grid_to_world(node.x, node.y) for node in grid_path]
        
        # Return both the simplified world path and the full grid path (for visualization)
        return world_path, grid_path

    def save_path_image(self, grid_path, output_path):
        """
        Draws the given grid_path onto the map and saves it to a file.
        
        :param grid_path: A list of 'GridNode' objects from the pathfinder.
        :param output_path: Full file path to save the new image.
        """
        print(f"Saving path visualization to {output_path}...")
        
        img = Image.fromarray(self.map_grid_raw).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        path_color = (255, 0, 0) # Red

        # Iterate over GridNode objects and use .x and .y
        for node in grid_path:
            x, y = node.x, node.y
            box = [x - 1, y - 1, x + 1, y + 1]
            draw.rectangle(box, fill=path_color)
            
        start_color = (0, 255, 0) # Green
        end_color = (0, 0, 255)   # Blue
        
        if grid_path:
            start_node = grid_path[0]
            end_node = grid_path[-1]
            
            start_box = [start_node.x - 2, start_node.y - 2, start_node.x + 2, start_node.y + 2]
            end_box = [end_node.x - 2, end_node.y - 2, end_node.x + 2, end_node.y + 2]
            
            draw.rectangle(start_box, fill=start_color)
            draw.rectangle(end_box, fill=end_color)
       
        try:
            img.save(output_path)
            print(f"Successfully saved {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")


# --- TEST SCRIPT ---
if __name__ == "__main__":
    """
    This block allows us to test the planner directly
    by running: python navigation_stack/planner/global_planner.py
    """
    print("--- Running GlobalPlanner Test ---")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLANNER_ROOT = SCRIPT_DIR
    STACK_ROOT = os.path.dirname(PLANNER_ROOT)
    PROJECT_ROOT = os.path.dirname(STACK_ROOT)
    
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    try:
        from navigation_stack.planner.global_planner import GlobalPlanner
    except ImportError as e:
        print(f"Error: {e}")
        print("Could not import GlobalPlanner. Make sure your __init__.py files are in place.")
        sys.exit(1)

    try:
        planner = GlobalPlanner()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure 'maps/map.yaml' and 'maps/map.png' exist inside 'navigation_stack/'")
        sys.exit(1)
    
    # We know the warehouse is at (0,0) and our map covers (-30 to 30)
    start_pos = (0.0, 0.0)    # Center of warehouse
    end_pos = (5.0, 5.0)   # A corner near the shelves  
    
    print(f"\nAttempting to plan from {start_pos} to {end_pos}")
    
    world_path, grid_path = planner.plan_path(start_pos, end_pos)
    
    
    if world_path:
        print(f"Length of path is : {len(world_path)}")
        print("\n--- Found Path (in world coordinates) ---")
        for i, waypoint in enumerate(world_path):
            if i % 10 == 0 or i == len(world_path) - 1:
                print(f"  {i}: ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
        
        output_dir = os.path.join(STACK_ROOT, "maps")
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, "path_visualization.png")
        
        planner.save_path_image(grid_path, output_image_path)
        
    else:
        print("\n--- Failed to find a path ---")
