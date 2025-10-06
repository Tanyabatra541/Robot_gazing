import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_box(origin, dimensions):
    """Create vertices for a 3D box."""
    x, y, z = origin
    dx, dy, dz = dimensions
    
    # Define the 8 vertices of the box
    vertices = [
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz]
    ]
    
    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # Top
    ]
    
    return faces

def rotate_vertices_z(vertices, angle_deg, center=(0,0,0)):
    """Rotate vertices around Z axis by angle_deg degrees."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy, cz = center
    rotated = []
    for v in vertices:
        x, y, z = v
        x -= cx
        y -= cy
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        rotated.append([x_new + cx, y_new + cy, z])
    return rotated

def rotate_vertices_x(vertices, angle_deg, center=(0,0,0)):
    """Rotate vertices around X axis by angle_deg degrees."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy, cz = center
    rotated = []
    for v in vertices:
        x, y, z = v
        y -= cy
        z -= cz
        y_new = y * cos_a - z * sin_a
        z_new = y * sin_a + z * cos_a
        rotated.append([x + cx, y_new + cy, z_new + cz])
    return rotated

def rotate_faces(faces, angle_deg, center=(0,0,0), axis='z'):
    if axis == 'z':
        return [rotate_vertices_z(face, angle_deg, center) for face in faces]
    elif axis == 'x':
        return [rotate_vertices_x(face, angle_deg, center) for face in faces]
    else:
        raise ValueError('Unsupported axis')

def create_robot(pan=30, tilt=0):
    """Create the robot components. pan: Z axis rotation, tilt: X axis rotation for head/eyes only."""
    robot_parts = []
    colors = []
    # Head (brown/tan color)
    head_faces = create_box([-2, -1, 2], [4, 2, 2])
    head_faces = rotate_faces(head_faces, pan, center=[0,0,3], axis='z')
    head_faces = rotate_faces(head_faces, tilt, center=[0,0,3], axis='x')
    robot_parts.extend(head_faces)
    colors.extend(['#B8860B'] * len(head_faces))
    # Left eye (blue)
    left_eye_faces = create_box([-1.5, -1.4, 2.8], [0.7, 0.4, 1])
    left_eye_faces = rotate_faces(left_eye_faces, pan, center=[0,0,3], axis='z')
    left_eye_faces = rotate_faces(left_eye_faces, tilt, center=[0,0,3], axis='x')
    robot_parts.extend(left_eye_faces)
    colors.extend(['#4A90E2'] * len(left_eye_faces))
    # Right eye (blue)
    right_eye_faces = create_box([0.8, -1.4, 2.8], [0.7, 0.4, 1])
    right_eye_faces = rotate_faces(right_eye_faces, pan, center=[0,0,3], axis='z')
    right_eye_faces = rotate_faces(right_eye_faces, tilt, center=[0,0,3], axis='x')
    robot_parts.extend(right_eye_faces)
    colors.extend(['#4A90E2'] * len(right_eye_faces))
    # Body (black)
    body_faces = create_box([-0.5, -0.5, -2], [1, 1, 4])
    robot_parts.extend(body_faces)
    colors.extend(['black'] * len(body_faces))
    return robot_parts, colors

def plot_robot(ax, pan = 0, tilt = 0):
    """Create an interactive 3D plot of the robot."""
    ax.cla()  # Clear the axes before plotting
    
    # Create robot with pan and tilt parameters
    robot_parts, colors = create_robot(pan=pan, tilt=tilt)
    
    # Add all robot parts to the plot
    face_collection = Poly3DCollection(robot_parts, 
                                       facecolors=colors, 
                                       edgecolors='black',
                                       linewidths=1,
                                       alpha=0.9)
    ax.add_collection3d(face_collection)
    
    # Set the aspect ratio and limits
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 5])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot', fontsize=14, pad=20)
    # Set initial view angle
    ax.view_init(elev=2, azim=-90)  # Front-facing view
    
    # Disable interactive navigation (mouse pan/tilt/zoom)
    ax.mouse_init()
    ax.disable_mouse_rotation()
    
    # Add grid
    ax.grid(False, alpha=0.0)
    
    plt.draw()
    plt.pause(0.001)
