import numpy as np

class CircularTrajectory3D:
    """
    Generate points on a 3D circle, tilted by two angles.
    """
    def __init__(self, center_x=0.0, center_y=0.0, center_z=0.0,
                 radius=1.0, tilt_theta=0.0, tilt_phi=0.0):
        self.center = np.array([center_x, center_y, center_z], dtype=np.float64)

        self.radius = float(radius)
        self.tilt_theta = float(tilt_theta)
        self.tilt_phi = float(tilt_phi)
        # Compute normal of the tilted circle plane
        nx = np.sin(self.tilt_theta) * np.cos(self.tilt_phi)
        ny = np.sin(self.tilt_theta) * np.sin(self.tilt_phi)
        nz = np.cos(self.tilt_theta)
        self.normal = np.array([nx, ny, nz], dtype=np.float64)
        self.normal /= np.linalg.norm(self.normal)
        # Choose a reference vector not parallel to normal
        if np.allclose(self.normal, [0, 0, 1.0], atol=1e-6):
            ref = np.array([0, 1, 0], dtype=np.float64)
        else:
            ref = np.array([0, 0, 1], dtype=np.float64)
        # In-plane axes
        self.u_axis = np.cross(self.normal, ref).astype(np.float64)
        self.u_axis /= np.linalg.norm(self.u_axis)
        self.v_axis = np.cross(self.normal, self.u_axis).astype(np.float64)
        self.v_axis /= np.linalg.norm(self.v_axis)

    def generate_trajectory(self, points=100):
        """
        Generate `points` positions (x,y,z) evenly around the circle.
        """
        angles = np.linspace(0, 2*np.pi, num=points, endpoint=False)
        coords = np.zeros((points, 3), dtype=np.float64)
        for i, theta in enumerate(angles):
            pt = (self.center +
                  self.radius * (np.cos(theta) * self.u_axis + np.sin(theta) * self.v_axis))
            coords[i, :] = pt
        x = coords[:, 0].astype(np.float32)
        y = coords[:, 1].astype(np.float32)
        z = coords[:, 2].astype(np.float32)
        return x, y, z
