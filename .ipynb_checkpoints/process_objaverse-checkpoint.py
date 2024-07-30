import os
import json
import trimesh
import pyrender
import numpy as np
import imageio
from PIL import Image

# Set up environment variables for rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['LD_PRELOAD'] = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'libstdc++.so.6')

# Define paths for input and output
input_dir = 'objaverse_data'
output_dir = 'objaverse_data_processed'
annotations_path = 'annotations/annotations.json'
processed_files_list_path = 'processed_glb_files.txt'
os.makedirs(output_dir, exist_ok=True)

# Load annotation data
with open(annotations_path, 'r') as file:
    annotations = json.load(file)

# attention: List all GLB files in the input directory, process only the first 100
glb_files = [f for f in os.listdir(input_dir) if f.endswith('.glb')][:100]

# Define views for rendering: horizontal rotation around the z-axis (middle vertical axis)
views = []
for angle in np.linspace(0, 360, 100):
    rad = np.deg2rad(angle)
    views.append({'position': [np.cos(rad) * 3, np.sin(rad) * 3, 0], 'target': [0, 0, 0]})


def look_at(eye, target, up=[0, 0, 1]):
    """Generate a look-at view matrix."""
    f = np.array(target) - np.array(eye)
    f = f / np.linalg.norm(f)
    u = np.array(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m

# Initialize a list to store successfully processed files
processed_files = []

for idx, glb_file in enumerate(glb_files):
    print(f"Processing file {idx + 1}/{len(glb_files)}: {glb_file}")
    glb_file_anno = glb_file.split('.')[0]
    
    # Check if the GLB file has annotations
    if glb_file_anno not in annotations:
        print(f"Skipping {glb_file_anno}: No annotations available.")
        continue
    
    # Load mesh from GLB file
    glb_path = os.path.join(input_dir, glb_file)
    mesh = trimesh.load(glb_path, force='mesh')

    # Retrieve dimensions for rendering
    try:
        width = int(annotations[glb_file_anno]['thumbnails']['images'][2]['width'])
        height = int(annotations[glb_file_anno]['thumbnails']['images'][2]['height'])
        print(f"Using dimensions: width = {width}, height = {height}")
    except (KeyError, IndexError) as e:
        print(f"Skipping {glb_file}: Invalid dimension data - {str(e)}")
        continue

    # Center and scale the mesh
    mesh.apply_translation(-mesh.centroid)
    scale_factor = 2 / max(mesh.extents)  #######change scale
    mesh.apply_scale(scale_factor)

    # Correct the orientation by applying a rotation
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    mesh.apply_transform(rotation_matrix)

    # Initialize renderer with specific dimensions
    # print(width)
    # print(height)
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    # Create a scene and add mesh
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    if isinstance(mesh, trimesh.Trimesh):
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)

    # Add lighting to the scene
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=np.eye(4))

    # Setup output directories
    uid = os.path.splitext(glb_file)[0]
    object_output_dir = os.path.join(output_dir, uid)
    os.makedirs(os.path.join(object_output_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(object_output_dir, 'pose'), exist_ok=True)

    # Render views and save images
    for view_idx, view in enumerate(views):
        camera_pose = look_at(view['position'], view['target'])
        camera = pyrender.IntrinsicsCamera(fx=450, fy=450, cx=width/2, cy=height/2, znear=0.1, zfar=100)
        scene_camera = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(scene_camera)
        color, depth = r.render(scene)
        
        # Ensure color has four channels (RGBA)
        alpha_channel = (depth != 0).astype(np.uint8) * 255 
        color = np.dstack((color, alpha_channel))

        # Center crop the image to height x height
        center_x = width // 2
        start_x = max(center_x - height // 2, 0)
        end_x = start_x + height
        cropped_color = color[:, start_x:end_x]

        # Resize the image to 512x512
        cropped_color = Image.fromarray(cropped_color, mode='RGBA')
        cropped_color = cropped_color.resize((512, 512), Image.LANCZOS)
        cropped_color = np.array(cropped_color)

        image_path = os.path.join(object_output_dir, 'rgb', f'{view_idx:03d}.png')
        imageio.imwrite(image_path, cropped_color)

        # Save camera pose for each view
        camera_pose_path = os.path.join(object_output_dir, 'pose', f'{view_idx:03d}.txt')
        np.savetxt(camera_pose_path, camera_pose)

        scene.remove_node(scene_camera)

    # Clean up renderer after processing
    r.delete()

    # Add the file name (without .glb extension) to the list of processed files
    with open(processed_files_list_path, 'a') as f:
        f.write(uid + '\n')

print("Processing completed.")
