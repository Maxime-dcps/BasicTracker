from filterpy.kalman import KalmanFilter
import numpy as np

def create_tracker():
    # dim_x = number of variables needed to describe the object; here (x, y, vx, vy)
    # dim_z = number of variables the detector will provide; here (x, y) center of BBox
    # TO DO : add height and width
    kf = KalmanFilter(dim_x=4, dim_z=2)

    dt = 1.0  # Delta t
    kf.F = np.array([[1, 0, dt, 0], # For x
                    [0, 1, 0, dt],  # For y
                    [0, 0, 1, 0],   # For vx
                    [0, 0, 0, 1]])  # For vy
    
    # Measurement matrix (H): we only measure x and y
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # Initial uncertainty (P)
    kf.P[2:, 2:] *= 1000.  # High uncertainty on initial velocities
    kf.P[:2, :2] *= 10.    # Moderate uncertainty on position

    # Measurement noise (R): confidence in the detector
    kf.R = np.array([[10, 0],
                     [0, 10]])

    # Process noise (Q): uncertainty in the motion model
    from filterpy.common import Q_discrete_white_noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=5.0, block_size=2)
    
    return kf