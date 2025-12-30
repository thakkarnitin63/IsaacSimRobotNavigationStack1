# # In: navigation_stack/task/map_generator.py
# # (REPLACE YOUR ENTIRE FILE)

# import sys
# import os
# import numpy as np
# import yaml
# import asyncio
# from PIL import Image

# # --- FIX PYTHON PATH ---
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# STACK_ROOT = os.path.dirname(SCRIPT_DIR)  # This is navigation_stack/
# PROJECT_ROOT = os.path.dirname(STACK_ROOT) # This is navigationStackv1/
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)
#     print(f"Project root added to path: {PROJECT_ROOT}")
# # -------------------------

# from isaacsim import SimulationApp

# # --- 2. LAUNCH ISAAC SIM ---
# simulation_app = SimulationApp({"headless": True})

# # --- 3. IMPORTS (must be after SimulationApp) ---
# import carb
# import omni
# import omni.physx
# from isaacsim.core.api import PhysicsContext
# from isaacsim.core.utils.stage import get_current_stage, open_stage_async
# from isaacsim.storage.native import get_assets_root_path
# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("isaacsim.asset.gen.omap")
# import omni.timeline
# from isaacsim.asset.gen.omap.bindings import _omap
# from pxr import UsdPhysics, Sdf, PhysxSchema


# class MapGenerator:
#     """
#     Generates a 2D occupancy map based on the official
#     Isaac Sim documentation and test files.
    
#     This version uses a reliable, hard-coded scan area
#     as all automatic bounds-detection methods have failed.
#     """
#     def __init__(self, maps_dir):
#         self._maps_dir = maps_dir
#         self._assets_root_path = get_assets_root_path()
#         if self._assets_root_path is None:
#             carb.log_error("Could not find Isaac Sim assets folder.")

#     async def run(self):
#         """
#         Loads the scene, warms up physics, generates the map, and saves it.
#         """
#         print("Starting in Map Generation Mode...")
#         if self._assets_root_path is None:
#             carb.log_error("Aborting, assets root path not found.")
#             return

#         # 1. Load Scene
#         scene_path = self._assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
#         await open_stage_async(scene_path)
#         stage = get_current_stage()
#         self.timeline = omni.timeline.get_timeline_interface()
        
#         # --- WAIT FOR STAGE TO LOAD ---
#         # (From the omni.kit.test example)
#         print("Waiting for stage to load...")
#         usd_context = omni.usd.get_context()
#         while usd_context.get_stage_loading_status()[2] > 0:
#             await omni.kit.app.get_app().next_update_async()
#         print("Stage loaded.")
#         # ------------------------------

#         # 2. Configure Physics (Robustly, from the omni.kit.test example)
#         self.physx_context = PhysicsContext(physics_dt=1.0 / 60.0)
        
#         UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
        
#         physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/PhysicsScene")
#         physxSceneAPI.CreateEnableStabilizationAttr(True)
#         physxSceneAPI.CreateSolverTypeAttr("TGS")
        
#         await omni.kit.app.get_app().next_update_async()  # Wait one frame for settings to apply

#         # 3. Start Physics and Warm Up
#         self.timeline.play()
        
#         print("Warming up physics engine (120 frames)...")
#         for _ in range(30): # Longer warmup for stability
#             await omni.kit.app.get_app().next_update_async()
        
#         print("Physics is stable.")

#         # 4. Generate Map (Following the documentation's example)
#         print("Configuring map generator...")
        
#         physx = omni.physx.get_physx_interface()
#         stage_id = omni.usd.get_context().get_stage_id()
#         generator = _omap.Generator(physx, stage_id)
        
#         # --- THIS IS THE ROBUST SOLUTION ---
#         # We will use hard-coded values, as shown in the documentation
#         # and test files. This avoids all errors from bounds calculation.
        
#         cell_size = 0.05
#         # Set output values: 1=occupied, 0=free, 255=unknown
#         generator.update_settings(cell_size, 1, 0, 255)

#         # Set a 60x60m scan box, 3m high, centered at the origin
#         # The scanner will start at (0,0,0.1) (the center)
#         map_origin = (0.0, 0.0, 0.1) # 10cm off the floor
#         min_bounds = (-15.0, -15.0, 0.0)
#         max_bounds = (15.0, 15.0, 3.0)
        
#         generator.set_transform(map_origin, min_bounds, max_bounds)
        
#         print(f"Using hard-coded bounds: {min_bounds} to {max_bounds}")
#         print(f"Using scanner origin: {map_origin}")
#         # --- END OF SOLUTION ---

#         print("Generating 2D occupancy map... This may take a moment.")
        
#         await omni.kit.app.get_app().next_update_async() # Apply settings
        
#         generator.generate2d()
        
#         print("Waiting for generator scan to complete (10 frames)...")
#         for _ in range(10):
#             await omni.kit.app.get_app().next_update_async()
        
#         buffer = generator.get_buffer()
#         dims = generator.get_dimensions()
            
#         print("Generation complete.")
        
#         # 5. Stop Physics
#         self.timeline.stop()
        
#         if not buffer or not dims or dims[0] == 0 or dims[1] == 0:
#             carb.log_error("Map generation failed. Buffer or dimensions are empty.")
#             return

#         width, height = dims[0], dims[1]
#         print(f"Map dimensions: {width} x {height}")

#         # Convert buffer to numpy array
#         buff_arr = np.array(buffer)

#        # Create an empty image array (initialized to Gray/Unknown)
#         # Standard ROS map: 205=Unknown (light gray) or 127=Unknown (dark gray)
#         image_data = np.full(buff_arr.shape, 127, dtype=np.uint8)

#         # Apply masks
#         # Free (0) -> White (255)
#         image_data[buff_arr == 0] = 255
        
#         # Occupied (1) -> Black (0)
#         image_data[buff_arr == 1] = 0
        
#         image_data = image_data.reshape((height, width))
#         img = Image.fromarray(image_data, mode='L')
#         os.makedirs(self._maps_dir, exist_ok=True)
#         map_image_path = os.path.join(self._maps_dir, "map.png")
#         img.save(map_image_path)
#         print(f"Saved map image to: {map_image_path}")
        
#         # 7. Save YAML File
#         map_yaml_path = os.path.join(self._maps_dir, "map.yaml")
#         map_min_bound = generator.get_min_bound()
#         yaml_data = {
#             "image": "map.png",
#             "resolution": cell_size,
#             "origin": [map_min_bound[0], map_min_bound[1], 0.0],
#             "negate": 0,
#             "occupied_thresh": 0.65,
#             "free_thresh": 0.196
#         }
#         with open(map_yaml_path, 'w') as f:
#             yaml.dump(yaml_data, f, default_flow_style=False)
#         print(f"Saved map YAML to: {map_yaml_path}")
#         print(f"--- NEW MAP ORIGIN: [{map_min_bound[0]}, {map_min_bound[1]}] ---")


# if __name__ == "__main__":
#     """
#     This block makes the async script runnable and properly reports exceptions.
#     """
#     # Save the map inside the 'navigation_stack' package
#     maps_dir = os.path.join(PROJECT_ROOT, "navigation_stack", "maps")
    
#     map_generator = MapGenerator(maps_dir)
#     task = None
#     try:
#         task = asyncio.ensure_future(map_generator.run())
#         while simulation_app.is_running() and not task.done():
#             simulation_app.update()
        
#         if task and task.exception():
#             raise task.exception()
            
#     except Exception as e:
#         print(f"An error occurred: {e}")
        
#     finally:
#         print("Closing simulation app...")
#         simulation_app.close()
#         sys.exit()




# In: navigation_stack/task/map_generator.py
import sys
import os
import numpy as np
import yaml
import asyncio
from PIL import Image

# --- FIX PYTHON PATH ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STACK_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(STACK_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -------------------------

from isaacsim import SimulationApp

# --- 2. LAUNCH ISAAC SIM ---
simulation_app = SimulationApp({"headless": True})

# --- 3. IMPORTS ---
import carb
import omni
import omni.physx
from isaacsim.core.api import PhysicsContext
from isaacsim.core.utils.stage import get_current_stage, open_stage_async
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.gen.omap")
import omni.timeline
from isaacsim.asset.gen.omap.bindings import _omap
from pxr import UsdPhysics, Sdf, PhysxSchema, UsdGeom, Gf, Usd

class MapGenerator:
    def __init__(self, maps_dir):
        self._maps_dir = maps_dir
        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder.")

    def _get_stage_bounds(self, stage):
        """
        Dynamically calculates the bounding box of the environment.
        """
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        combined_range = Gf.Range3d()

        root_prim = stage.GetPseudoRoot()
        for prim in root_prim.GetChildren():
            if prim.GetName() in ["Environment", "Skybox", "Cameras", "Lights", "Looks", "Render"]:
                continue
            if not prim.IsA(UsdGeom.Imageable):
                continue

            bound = bbox_cache.ComputeWorldBound(prim)
            prim_range = bound.ComputeAlignedBox()
            if not prim_range.IsEmpty():
                combined_range.UnionWith(prim_range)

        if combined_range.IsEmpty():
            print("⚠️ Warning: Could not auto-detect bounds. Using defaults.")
            return (-20.0, -20.0, 0.0), (20.0, 20.0, 3.0)

        min_box = combined_range.GetMin()
        max_box = combined_range.GetMax()
        margin = 2.0
        return (min_box[0]-margin, min_box[1]-margin, 0.0), (max_box[0]+margin, max_box[1]+margin, 3.0)

    async def run(self):
        print("Starting in Map Generation Mode...")
        if self._assets_root_path is None:
            carb.log_error("Aborting, assets root path not found.")
            return

        # 1. Load Scene
        # Using the warehouse map that we know works
        scene_path = self._assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        await open_stage_async(scene_path)
        
        stage = get_current_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        
        # Wait for Stage to load
        print("Waiting for stage to load...")
        usd_context = omni.usd.get_context()
        while usd_context.get_stage_loading_status()[2] > 0:
            await omni.kit.app.get_app().next_update_async()
        print("Stage loaded.")
        # ------------------------------

        # 2. Configure Physics
        self.physx_context = PhysicsContext(physics_dt=1.0 / 60.0)
        UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/PhysicsScene")
        physxSceneAPI.CreateEnableStabilizationAttr(True)
        physxSceneAPI.CreateSolverTypeAttr("TGS")
        await omni.kit.app.get_app().next_update_async()

        # 3. Warm Up
        self.timeline.play()
        print("Warming up physics engine (60 frames)...")
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        # 4. Generate Map
        print("Configuring map generator...")
        physx = omni.physx.get_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        generator = _omap.Generator(physx, stage_id)

        # --- DYNAMIC BOUNDS CALCULATION (The new part) ---
        print("Auto-calculating map bounds...")
        min_bounds, max_bounds = self._get_stage_bounds(stage)
        map_origin = (0.0, 0.0, 0.1)
        
        print(f"Detected Bounds: {min_bounds} -> {max_bounds}")

        # 0.05 resolution (5cm)
        generator.update_settings(0.05, 1, 0, 255)
        generator.set_transform(map_origin, min_bounds, max_bounds)
        # -------------------------------------------------

        print("Generating 2D occupancy map...")
        await omni.kit.app.get_app().next_update_async()
        generator.generate2d()
        
        print("Waiting for generator scan to complete (10 frames)...")
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        
        buffer = generator.get_buffer()
        dims = generator.get_dimensions()
        self.timeline.stop()
        
        if not buffer:
            carb.log_error("Map generation failed.")
            return

        width, height = dims[0], dims[1]
        print(f"Map dimensions: {width} x {height}")

        # 5. Save PNG (Bug Fixed Version)
        buff_arr = np.array(buffer, dtype=np.uint8)
        
        # Init to Gray (Unknown)
        image_data = np.full(buff_arr.shape, 127, dtype=np.uint8)
        # Free -> White
        image_data[buff_arr == 0] = 255
        # Occupied -> Black
        image_data[buff_arr == 1] = 0
        
        image_data = image_data.reshape((height, width))
        img = Image.fromarray(image_data)
        
        os.makedirs(self._maps_dir, exist_ok=True)
        img.save(os.path.join(self._maps_dir, "map.png"))
        print(f"Saved map.png")
        
        # 6. Save YAML
        map_yaml_path = os.path.join(self._maps_dir, "map.yaml")
        min_b = generator.get_min_bound()
        yaml_data = {
            "image": "map.png",
            "resolution": 0.05,
            "origin": [min_b[0], min_b[1], 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196
        }
        with open(map_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        print(f"Saved map.yaml")


if __name__ == "__main__":
    maps_dir = os.path.join(PROJECT_ROOT, "navigation_stack", "maps")
    map_generator = MapGenerator(maps_dir)
    task = None
    try:
        task = asyncio.ensure_future(map_generator.run())
        while simulation_app.is_running() and not task.done():
            simulation_app.update()
        if task and task.exception():
            raise task.exception()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing simulation app...")
        simulation_app.close()
        sys.exit()