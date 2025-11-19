import open3d as o3d
import numpy as np

def main():
    # 1. Load the points from the file
    try:
        points = np.load("first_scan.npy")
    except FileNotFoundError:
        print("Error: 'first_scan.npy' not found.")
        print("Please run the Isaac Sim script first to generate the file.")
        return
        
    print(f"Loaded {points.shape[0]} points.")

    # 2. Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # 3. Assign the points from your NumPy array
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 4. (Optional) Color the points green
    pcd.paint_uniform_color([0.0, 1.0, 0.0])
    
    # 5. (Crucial) Create a world coordinate frame to see the origin (0,0,0)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )

    # 6. Show the point cloud and the world frame
    print("Showing point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries([pcd, world_frame])

if __name__ == "__main__":
    main()