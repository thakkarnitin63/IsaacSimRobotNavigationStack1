import carb
import sys
import numpy as np
import omni
from pxr import UsdGeom, Gf
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.rtx import LidarRtx
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from scipy.spatial.transform import Rotation as R
from isaacsim.core.utils.rotations import quat_to_rot_matrix


class NovaCarter:
    """
    This class is responsible for spawning and initializing the 
    Nova Carter robot and its sensors in the Isaac Sim stage.
    
    It assumes the main environment/scene USD is already open.
    """
    def __init__(
        self,
        robot_prim_path: str = "/World/Nova_Carter",
        robot_name: str = "my_carter",
        position: np.ndarray = np.array([0, 0, 0.1]),
    ):
        """
        Args:
            robot_prim_path (str): The prim path to spawn the robot at.
            robot_name (str): The name to give the robot object.
            position (np.ndarray): The [x, y, z] position to spawn the robot.
        """
        self.robot_prim_path = robot_prim_path
        self.robot_name = robot_name
        self.position = position
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder. Check Nucleus connection.")
            sys.exit()

        # This class only cares about the ROBOT'S USD
        self.robot_usd_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
        
        self._robot = None
        self._lidar_vis = None

        self._controller = DifferentialController(name="simple_control",
                                                  wheel_radius=0.04295, 
                                                  wheel_base=0.4132)
        self._debug_counter = 0
        # self.annotator_key = "IsaacExtractRTXSensorPointCloudNoAccumulator"
        self.annotator_key = "IsaacCreateRTXLidarScanBuffer"

    def spawn(self):
        """
        Spawns the robot USD and creates the WheeledRobot and LidarRtx objects.
        Assumes a stage is already open.
        """
        self._robot = WheeledRobot(
            prim_path=self.robot_prim_path,
            name=self.robot_name,
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=self.robot_usd_path,
            position=self.position,
        )
        
        # This prim path is relative to the robot's prim path
        lidar_prim_path = self.robot_prim_path + "/chassis_link/sensors/XT_32/PandarXT_32_10hz"
        
        self._lidar_vis = LidarRtx(
            prim_path=lidar_prim_path,  
            name="lidar_visualizer",
            config='HESAI_XT32_SD10'

        )
        
        omni.kit.app.get_app().update()
        print(f"Robot '{self.robot_name}' spawned at '{self.robot_prim_path}'.")

    def initialize(self):
        """
        Initializes the Lidar visualizer and the robot's physics.
        """
        if self._robot is None or self._lidar_vis is None:
            carb.log_error("Robot must be spawned before initializing.")
            return

        print("Initializing Lidar visualizer...")
        self._lidar_vis.initialize()
        self._lidar_vis.attach_annotator(self.annotator_key)
        # self._lidar_vis.enable_visualization()
        print("Lidar visualizer enabled.")
        
        self._robot.initialize()
        print("Robot initialized.")

    def get_world_pose(self):
        """
        Returns the robot's current position and orientation in the world frame.

        Returns:
            (np.ndarray, np.ndarray): (position, orientation_quaternion)
        """
        return self._robot.get_world_pose()

    def get_linear_velocity(self):
        """
        Returns the robot's current linear velocity in the world frame.

        Returns:
            np.ndarray: (vx, vy, vz)
        """
        return self._robot.get_linear_velocity()

    def get_angular_velocity(self):
        """
        Returns the robot's current angular velocity in the world frame.

        Returns:
            np.ndarray: (wx, wy, wz)
        """
        return self._robot.get_angular_velocity()
    
    def get_lidar_data(self):
        """
        Returns the latest point cloud data from the Lidar.

        Returns:
            dict: A dictionary containing the point cloud data.
        """
        return self._lidar_vis.get_current_frame()
    
    def apply_drive_commands(self, linear_velocity: float, angular_velocity: float):
        """
        Applies linear and angular velocity commands to the robot's wheels.
        Uses direct differential drive kinematics for debugging.

        Args:
            linear_velocity (float): Forward speed in m/s.
            angular_velocity (float): Rotational speed in rad/s.
        """
        # Robot parameters
        wheel_radius = 0.04295  # meters
        wheel_base = 0.4132     # meters (distance between wheels)
        
        # Direct differential drive kinematics
        # v_left = v - (ω * L / 2)
        # v_right = v + (ω * L / 2)
        v_left = linear_velocity - (angular_velocity * wheel_base / 2.0)
        v_right = linear_velocity + (angular_velocity * wheel_base / 2.0)
        
        # Convert linear wheel velocities to angular velocities (rad/s)
        omega_left = v_left / wheel_radius
        omega_right = v_right / wheel_radius
        
        # Debug output every 100 calls
        self._debug_counter += 1
        # if self._debug_counter % 100 == 0:
        #     print(f"  [DRIVE] Input: v={linear_velocity:.3f} m/s, ω={angular_velocity:.3f} rad/s")
        #     print(f"  [DRIVE] Wheel speeds: left={omega_left:.2f}, right={omega_right:.2f} rad/s")
        actions = self._controller.forward(command=[linear_velocity, angular_velocity])
        #  Debug: Print what the controller outputs
        # if self._debug_counter % 100 == 0:
        #     print(f"  [DRIVE] Controller output type: {type(actions)}")
        #     print(f"  [DRIVE] Controller output: {actions}")
        
        # Apply the actions to the robot
        self._robot.apply_wheel_actions(actions)

    def get_lidar_to_base_transform(self):
        """
        Gets the static transform (position, orientation) of the Lidar sensor
        relative to the robot's base prim.
        
        Returns:
            (np.ndarray, np.ndarray): (position, orientation_quat_xyzw)
            Returns (None, None) if prims are not found.
        """
        try:
            # 1. Get the prim for the robot base
            base_prim = self._robot.prim
            
            # 2. Get the prim for the Lidar
            lidar_prim_path = self.robot_prim_path + "/chassis_link/sensors/XT_32/PandarXT_32_10hz"
            lidar_prim = omni.usd.get_context().get_stage().GetPrimAtPath(lidar_prim_path)
            
            if not base_prim:
                carb.log_error("Could not find robot base prim.")
                return None, None
            if not lidar_prim:
                carb.log_error(f"Could not find Lidar prim at {lidar_prim_path}")
                return None, None
                
            # 3. Get the world transforms of both prims
            base_world_transform = omni.usd.get_world_transform_matrix(base_prim)
            lidar_world_transform = omni.usd.get_world_transform_matrix(lidar_prim)
            
            # 4. Calculate the Lidar's pose relative to the base
            # T(lidar_to_base) = T(world_to_base) * T(lidar_to_world)
            # T(world_to_base) = T(base_to_world).GetInverse()
            base_inv_transform = base_world_transform.GetInverse()
            lidar_to_base_transform = base_inv_transform * lidar_world_transform
            
            # 5. Decompose the transform into pos and quat
            pos = lidar_to_base_transform.ExtractTranslation()
            quat = lidar_to_base_transform.ExtractRotationQuat()
            
            # 6. Convert to numpy arrays
            pos_np = np.array(pos)
            
            # Convert Gf.Quat (w, i, j, k) to (x, y, z, w) format
            quat_np = np.array([quat.GetImaginary()[0], 
                                quat.GetImaginary()[1], 
                                quat.GetImaginary()[2], 
                                quat.GetReal()])
            
            return pos_np, quat_np
            
        except Exception as e:
            carb.log_error(f"Error in get_lidar_to_base_transform: {e}")
            return None, None
        

    # def _transform_local_points_to_world(self, points_local):
    #     """
    #     Transforms local Lidar points to world frame.
    #     Uses Isaac Sim's native transformation - more reliable.
    #     """
    #     if points_local.size == 0:
    #         return np.array([])

    #     try:
    #         # Get the lidar prim's world transform matrix
    #         stage = omni.usd.get_context().get_stage()
    #         lidar_prim_path = self.robot_prim_path + "/chassis_link/sensors/XT_32/PandarXT_32_10hz"
    #         lidar_prim = stage.GetPrimAtPath(lidar_prim_path)
            
    #         if not lidar_prim:
    #             carb.log_error("Could not find lidar prim for transformation")
    #             return np.array([])
            
    #         # Get 4x4 world transform matrix
    #         world_matrix = omni.usd.get_world_transform_matrix(lidar_prim)
            
    #         # Extract 3x3 rotation and translation
    #         # USD matrix is column-major, we need row-major for numpy
    #         rotation_matrix = np.array([
    #             [world_matrix[0][0], world_matrix[1][0], world_matrix[2][0]],
    #             [world_matrix[0][1], world_matrix[1][1], world_matrix[2][1]],
    #             [world_matrix[0][2], world_matrix[1][2], world_matrix[2][2]]
    #         ])
            
    #         translation = np.array([
    #             world_matrix[3][0],
    #             world_matrix[3][1],
    #             world_matrix[3][2]
    #         ])
            
    #         # Apply transformation: P_world = R * P_local + T
    #         points_world = (rotation_matrix @ points_local.T).T + translation
            
    #         return points_world
            
    #     except Exception as e:
    #         carb.log_error(f"Transform error: {e}")
    #         return np.array([])



    def _transform_local_points_to_world(self, points_local):
        """
        Transforms local Lidar points to world frame.
        Uses Isaac Sim's quat_to_rot_matrix with the CORRECT quaternion format.
        
        Args:
            points_local: Nx3 array of points in lidar's local frame
            
        Returns:
            Nx3 array of points in world frame
        """
        if points_local.size == 0:
            return np.array([])

        try:
            # Get lidar's world pose
            transform_matrix = self.get_sensor_pose_matrix()
        
            # Extract rotation and translation
            rotation_matrix = transform_matrix[0:3, 0:3]
            translation = transform_matrix[0:3, 3]
            
            # Apply transformation: P_world = R @ P_local + T
            points_world = (rotation_matrix @ points_local.T).T + translation
            
            return points_world
            
        except Exception as e:
            carb.log_error(f"Lidar transform error: {e}")
            return np.array([])
        
    def get_sensor_pose_matrix(self):
        """
        Get the lidar sensor's pose as a 4x4 transformation matrix in world frame.
        
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix (sensor-to-world)
                    [[R11, R12, R13, Tx],
                        [R21, R22, R23, Ty],
                        [R31, R32, R33, Tz],
                        [  0,   0,   0,  1]]
        """
        # Get position and quaternion from lidar
        lidar_pos, lidar_quat_xyzw = self._lidar_vis.get_world_pose()
        
        # Convert quaternion to rotation matrix
        # Isaac Sim's quat_to_rot_matrix expects [x, y, z, w] format
        rotation_matrix = quat_to_rot_matrix(lidar_quat_xyzw)
        
        # Build 4x4 homogeneous transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rotation_matrix  # Rotation part
        transform_matrix[0:3, 3] = lidar_pos          # Translation part
        
        return transform_matrix
    
    def get_robot_pose_vector(self):
        """
        Get the robot base position in world frame.
        This is used by STVL to center the voxel grid on the robot.
        
        Returns:
            np.ndarray: [x, y, z] position in world frame
        
        Example output:
            array([10.0, 5.0, 0.0])  # Robot at (10m, 5m, 0m) in world
        """
        robot_pos, _ = self.get_world_pose()
        return robot_pos


    def get_lidar_world_points(self):
        """
        Gets lidar scan and transforms to world frame.
        Combines get_lidar_points_in_sensor_frame() + _transform_local_points_to_world()
        """
        points_local = self.get_lidar_points_in_sensor_frame()
        
        if points_local.size == 0:
            return np.array([])
        
        # Debug output (optional)
        if self._debug_counter % 200 == 0:
            print(f"\n🔍 LIDAR COORDINATE DEBUG:")
            print(f"   Annotator: {self.annotator_key}")
            print(f"   Sample local point: {points_local[0]}")
            
            lidar_pos, lidar_quat = self._lidar_vis.get_world_pose()
            print(f"   Lidar world pos: {lidar_pos}")
            print(f"   Lidar world quat (xyzw): {lidar_quat}")
            
            robot_pos = self.get_robot_pose_vector()
            print(f"   Robot world pos: {robot_pos}")
            
            dist = np.linalg.norm(points_local[0])
            print(f"   Distance to first point: {dist:.2f}m")
        
        return self._transform_local_points_to_world(points_local)
    
    def get_lidar_points_in_sensor_frame(self):
        """
        Get raw lidar points in the sensor's local frame.
        
        Returns:
            np.ndarray: [N, 3] array of points in sensor frame
                    Returns empty array if no data available
        
        Example output:
            array([[1.5, 0.2, 0.1],
                [2.3, -0.5, 0.3],
                ...])
        """
        lidar_raw_data = self.get_lidar_data()
        
        if (lidar_raw_data and 
            self.annotator_key in lidar_raw_data and 
            'data' in lidar_raw_data[self.annotator_key]):
            
            points_local = lidar_raw_data[self.annotator_key]['data']
            return points_local
        
        return np.array([])
    

    def debug_lidar_transform(self):
        """
        Debug helper to print and verify lidar transformation.
        Call this once to check if everything is set up correctly.
        """
        print("\n" + "="*70)
        print("LIDAR TRANSFORMATION VERIFICATION")
        print("="*70)
        
        # 1. Get robot's world pose
        robot_pos, robot_quat = self.get_world_pose()
        print(f"\n📍 Robot Base (World Frame):")
        print(f"   Position: {robot_pos}")
        print(f"   Quaternion: {robot_quat}")
        
        # 2. Get lidar's world pose (direct)
        try:
            lidar_pos, lidar_quat = self._lidar_vis.get_world_pose()
            print(f"\n📡 Lidar Sensor (World Frame - Direct):")
            print(f"   Position: {lidar_pos}")
            print(f"   Quaternion: {lidar_quat}")
        except:
            print("\n❌ Could not get lidar world pose!")
            return
        
        # 3. Get the static transform
        lidar_to_base_pos, lidar_to_base_quat = self.get_lidar_to_base_transform()
        print(f"\n🔗 Lidar-to-Base Transform (Static):")
        print(f"   Offset Position: {lidar_to_base_pos}")
        print(f"   Offset Quaternion: {lidar_to_base_quat}")
        
        # 4. Get sample lidar data
        lidar_data = self.get_lidar_data()
        if not lidar_data or self.annotator_key not in lidar_data:
            print("\n❌ No lidar data available yet!")
            return
            
        points_local = lidar_data[self.annotator_key]['data']
        if points_local.size == 0:
            print("\n❌ Lidar data is empty!")
            return
        
        print(f"\n📊 Sample Lidar Points:")
        print(f"   Total points: {len(points_local)}")
        print(f"   First 3 points (Lidar Frame):")
        for i in range(min(3, len(points_local))):
            print(f"      [{i}]: {points_local[i]}")
        
        # 5. Transform to world
        points_world = self._transform_local_points_to_world(points_local)
        print(f"\n   First 3 points (World Frame):")
        for i in range(min(3, len(points_world))):
            print(f"      [{i}]: {points_world[i]}")
        
        # 6. Calculate expected position of first point manually
        print(f"\n🔍 Manual Verification of First Point:")
        first_local = points_local[0]
        print(f"   Local point: {first_local}")
        
        # Apply transformation manually
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat(lidar_quat)
        first_rotated = rot.apply(first_local)
        first_world_manual = first_rotated + lidar_pos
        
        print(f"   After rotation: {first_rotated}")
        print(f"   After translation: {first_world_manual}")
        print(f"   Function output: {points_world[0]}")
        print(f"   ✓ Match: {np.allclose(first_world_manual, points_world[0])}")
        
        # 7. Sanity checks
        print(f"\n✅ Sanity Checks:")
        
        # Check 1: Lidar should be above robot
        if lidar_pos[2] > robot_pos[2]:
            print(f"   ✓ Lidar Z ({lidar_pos[2]:.3f}) > Robot Z ({robot_pos[2]:.3f})")
        else:
            print(f"   ❌ Lidar should be above robot!")
        
        # Check 2: World points should be near robot
        mean_distance = np.linalg.norm(points_world[:, :2] - robot_pos[:2], axis=1).mean()
        print(f"   ✓ Mean distance to robot: {mean_distance:.2f}m")
        
        if mean_distance < 20:
            print(f"   ✓ Points are reasonably close (< 20m)")
        else:
            print(f"   ⚠️ Points seem far away (> 20m)")
        
        # Check 3: Z-range in world frame
        z_min, z_max = points_world[:, 2].min(), points_world[:, 2].max()
        print(f"   ✓ World Z-range: [{z_min:.2f}, {z_max:.2f}]m")
        
        if z_min < -1.0:
            print(f"   ⚠️ Some points are below ground (< -1m)")
        
        print("="*70 + "\n")


        print(f"\n🧪 Quaternion Conversion Test:")
        _, quat_xyzw = self._lidar_vis.get_world_pose()
        x, y, z, w = quat_xyzw
        quat_wxyz = np.array([w, x, y, z])
        
        print(f"   Input (xyzw): {quat_xyzw}")
        print(f"   Reordered (wxyz): {quat_wxyz}")
        
        rot_mat = quat_to_rot_matrix(quat_wxyz)
        print(f"   Rotation matrix shape: {rot_mat.shape}")
        print(f"   Matrix determinant: {np.linalg.det(rot_mat):.6f} (should be 1.0)")
        
        # Test with identity quaternion
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        identity_matrix = quat_to_rot_matrix(identity_quat)
        print(f"   Identity test (should be eye(3)): {np.allclose(identity_matrix, np.eye(3))}")


    def debug_transformation_comparison(self):
        """
        Compares USD matrix method vs quaternion method to find the discrepancy.
        This will reveal exactly what's wrong with the quaternion approach.
        """
        print("\n" + "="*80)
        print("🔬 TRANSFORMATION COMPARISON DEBUG")
        print("="*80)
        
        try:
            # Get both the quaternion and the USD matrix
            lidar_pos_quat, lidar_quat_xyzw = self._lidar_vis.get_world_pose()
            
            stage = omni.usd.get_context().get_stage()
            lidar_prim_path = self.robot_prim_path + "/chassis_link/sensors/XT_32/PandarXT_32_10hz"
            lidar_prim = stage.GetPrimAtPath(lidar_prim_path)
            world_matrix = omni.usd.get_world_transform_matrix(lidar_prim)
            
            print(f"\n📍 From get_world_pose():")
            print(f"   Position: {lidar_pos_quat}")
            print(f"   Quaternion [x,y,z,w]: {lidar_quat_xyzw}")
            
            print(f"\n📍 From USD Matrix:")
            translation_usd = np.array([world_matrix[3][0], world_matrix[3][1], world_matrix[3][2]])
            print(f"   Position: {translation_usd}")
            
            # Extract rotation matrix from USD
            rotation_usd = np.array([
                [world_matrix[0][0], world_matrix[1][0], world_matrix[2][0]],
                [world_matrix[0][1], world_matrix[1][1], world_matrix[2][1]],
                [world_matrix[0][2], world_matrix[1][2], world_matrix[2][2]]
            ])
            
            print(f"\n🔄 USD Rotation Matrix:")
            print(rotation_usd)
            print(f"   Determinant: {np.linalg.det(rotation_usd):.6f} (should be 1.0)")
            
            # Try Isaac Sim's converter
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            x, y, z, w = lidar_quat_xyzw
            quat_wxyz = np.array([w, x, y, z])
            rotation_isaac = quat_to_rot_matrix(quat_wxyz)
            
            print(f"\n🔄 Isaac Sim quat_to_rot_matrix [w,x,y,z]:")
            print(rotation_isaac)
            print(f"   Determinant: {np.linalg.det(rotation_isaac):.6f}")
            
            # Check if matrices match
            print(f"\n✅ Matrix Comparison:")
            print(f"   USD vs Isaac Sim match: {np.allclose(rotation_usd, rotation_isaac, atol=1e-5)}")
            print(f"   Max difference: {np.abs(rotation_usd - rotation_isaac).max():.10f}")
            
            # Try manual quaternion to matrix
            rotation_manual = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])
            
            print(f"\n🔄 Manual Quaternion Formula [x,y,z,w]:")
            print(rotation_manual)
            print(f"   USD vs Manual match: {np.allclose(rotation_usd, rotation_manual, atol=1e-5)}")
            print(f"   Max difference: {np.abs(rotation_usd - rotation_manual).max():.10f}")
            
            # Test transformation with a sample point
            test_point = np.array([1.0, 0.0, 0.0])  # 1 meter in X direction
            
            print(f"\n🧪 Test Point Transformation [1, 0, 0]:")
            point_usd = rotation_usd @ test_point + translation_usd
            point_isaac = rotation_isaac @ test_point + lidar_pos_quat
            point_manual = rotation_manual @ test_point + lidar_pos_quat
            
            print(f"   USD result: {point_usd}")
            print(f"   Isaac Sim result: {point_isaac}")
            print(f"   Manual result: {point_manual}")
            
            print(f"\n   Match USD vs Isaac: {np.allclose(point_usd, point_isaac, atol=1e-5)}")
            print(f"   Match USD vs Manual: {np.allclose(point_usd, point_manual, atol=1e-5)}")
            
            # Check position mismatch
            print(f"\n📏 Position Comparison:")
            print(f"   get_world_pose(): {lidar_pos_quat}")
            print(f"   USD matrix:       {translation_usd}")
            print(f"   Position match: {np.allclose(lidar_pos_quat, translation_usd, atol=1e-5)}")
            print(f"   Position diff: {np.abs(lidar_pos_quat - translation_usd).max():.10f}")
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"❌ Debug error: {e}")
    def debug_find_axis_transform(self):
        """Try to find the coordinate transformation between conventions."""
        print("\n🔬 Searching for axis transformation...")
        
        lidar_pos, lidar_quat_xyzw = self._lidar_vis.get_world_pose()
        
        stage = omni.usd.get_context().get_stage()
        lidar_prim_path = self.robot_prim_path + "/chassis_link/sensors/XT_32/PandarXT_32_10hz"
        lidar_prim = stage.GetPrimAtPath(lidar_prim_path)
        world_matrix = omni.usd.get_world_transform_matrix(lidar_prim)
        
        rotation_usd = np.array([
            [world_matrix[0][0], world_matrix[1][0], world_matrix[2][0]],
            [world_matrix[0][1], world_matrix[1][1], world_matrix[2][1]],
            [world_matrix[0][2], world_matrix[1][2], world_matrix[2][2]]
        ])
        
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        x, y, z, w = lidar_quat_xyzw
        
        # Try different quaternion orderings
        orderings = {
            "[w,x,y,z]": np.array([w, x, y, z]),
            "[x,y,z,w]": np.array([x, y, z, w]),
            "[w,z,y,x]": np.array([w, z, y, x]),
            "[w,y,z,x]": np.array([w, y, z, x]),
            "[w,-x,-y,z]": np.array([w, -x, -y, z]),
        }
        
        for name, quat in orderings.items():
            try:
                rot = quat_to_rot_matrix(quat)
                diff = np.abs(rotation_usd - rot).max()
                print(f"   {name}: max diff = {diff:.6f}")
                if diff < 0.001:
                    print(f"   ✅ MATCH FOUND: {name}")
                    print(f"   Use: quat = np.array({quat.tolist()})")
            except:
                pass