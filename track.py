from filter import create_tracker
import numpy as np

class Track:
    def __init__(self, first_detection, track_id):
        self.id = track_id
        self.kf = create_tracker()
        
        # Convert BBox to [cx, cy, w, h] for the filter
        x1, y1, x2, y2 = first_detection
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.kf.x[:2] = np.array([[cx], [cy]])
        self.width = w    # Store the width
        self.height = h   # Store the height
        
        self.age = 0
        self.hits = 1
        self.no_losses = 0

    def update(self, detection):
        self.no_losses = 0
        self.hits += 1
        
        x1, y1, x2, y2 = detection
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.width = x2 - x1
        self.height = y2 - y1
        
        z = np.array([[cx], [cy]])
        # Update Kalman filter with BBox center
        self.kf.update(z)

    def mark_missed(self):
        self.no_losses += 1

    def get_bbox(self):
        cx, cy = self.kf.x[0, 0], self.kf.x[1, 0]
        x1 = cx - self.width / 2
        y1 = cy - self.height / 2
        x2 = cx + self.width / 2
        y2 = cy + self.height / 2
        return [x1, y1, x2, y2]

    @property
    def position(self):
        return (int(self.kf.x[0, 0]), int(self.kf.x[1, 0]))