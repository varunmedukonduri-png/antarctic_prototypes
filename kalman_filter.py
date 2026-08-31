import numpy as np

class KalmanFilter2D:
    """
    Simple 2D Kalman filter for smoothing position estimates.
    State: [x, y, vx, vy]  (latitude, longitude, velocity in lat/lon per step)
    """
    def __init__(self, dt=1.0, process_noise=1e-3, measurement_noise=1e-1):
        self.dt = dt
        # State transition matrix (constant velocity)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        # Initial state and covariance
        self.x = None
        self.P = None

    def initialize(self, x, y):
        """Initialize the filter with the first measurement."""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 100.0  # high uncertainty initially

    def predict(self):
        """Predict the next state."""
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]  # return predicted position (lat, lon)

    def update(self, z):
        """Update the filter with a new measurement z = [lat, lon]."""
        if self.x is None:
            self.initialize(z[0], z[1])
            return z
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]

def apply_kalman_to_path(raw_path, process_noise=1e-3, measurement_noise=1e-1):
    """
    Apply Kalman filter to a list of (lat, lon) points.
    Returns filtered path as list of (lat, lon).
    """
    if not raw_path:
        return []
    kf = KalmanFilter2D(process_noise=process_noise, measurement_noise=measurement_noise)
    filtered = []
    for i, (lat, lon) in enumerate(raw_path):
        if i == 0:
            kf.initialize(lat, lon)
            filtered.append((lat, lon))
        else:
            # Predict then update
            kf.predict()
            filtered_pos = kf.update(np.array([lat, lon]))
            filtered.append((filtered_pos[0], filtered_pos[1]))
    return filtered
