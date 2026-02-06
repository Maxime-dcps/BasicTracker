from filterpy.kalman import KalmanFilter
import numpy as np

def create_tracker():
    # dim_x = number of variables needed to describe the object; here (x, y, h, r vx, vy, vh, vr)
    # dim_z = number of variables the detector will provide; here (x, y, h, r) center of BBox
    kf = KalmanFilter(dim_x=8, dim_z=4)

    dt = 1.0  # Delta t
    kf.F = np.array([
                        [1, 0, 0, 0, dt, 0, 0, 0],  # For x
                        [0, 1, 0, 0, 0, dt, 0, 0],  # For y
                        [0, 0, 1, 0, 0, 0, dt, 0],  # For h
                        [0, 0, 0, 1, 0, 0, 0, dt],  # For r
                        [0, 0, 0, 0, 1, 0, 0, 0],   # For vx
                        [0, 0, 0, 0, 0, 1, 0, 0],   # For vy
                        [0, 0, 0, 0, 0, 0, 1, 0],   # For vh
                        [0, 0, 0, 0, 0, 0, 0, 0],   # For vr
                    ])
    
    # Measurement matrix (H)
    kf.H = np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0]])

    # Initial uncertainty (P)
    kf.P[4:, 4:] *= 1000.  # High uncertainty on initial velocities
    kf.P[:4, :4] *= 10.    # Moderate uncertainty on position

    # Measurement noise (R): confidence in the detector
    kf.R = np.array([[10, 0, 0, 0],
                     [0, 10, 0, 0],
                     [0, 0, 50, 0],
                     [0, 0, 0, 70]])

    # Process noise (Q): uncertainty in the motion model
    from filterpy.common import Q_discrete_white_noise, block_diag
    qxy = Q_discrete_white_noise(dim=2, dt=dt, var=5.0)
    qhw = Q_discrete_white_noise(dim=2, dt=dt, var=0.5)
    kf.Q = block_diag(qxy, qxy, qhw, qhw)
    
    return kf