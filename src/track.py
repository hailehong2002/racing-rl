import numpy as np
from scipy.interpolate import splprep, splev
import f1
from src.config import (
    YEAR,
    GRAND_PRIX,
    SESSION_TYPE,
    DRIVERS,
    CACHE_DIR,
    N_POINTS
)

class Track:
    def __init__(self, x, y, n_points: int = N_POINTS):
        self.n_points = n_points
        self.waypoints = self._from_coords(x, y, n_points)
        self.segment_lengths = self._compute_segment_lengths()
        self.total_length = self.segment_lengths.sum()
        
    def _from_coords(self, x, y, n_points):  # Interpolate the track coordinates to get a smooth path using B-splines
        tck, _ = splprep([x, y], s=0, per=True)
        u = np.linspace(0, 1, n_points, endpoint=False)
        x_smoothed, y_smoothed = splev(u, tck)
        return np.stack([x_smoothed, y_smoothed], axis=1)

    def _compute_segment_lengths(self) -> np.ndarray:   #Compute the lengths of each segment between waypoints
    diff = np.roll(self.waypoints, -1, axis=0) - self.waypoints
    return np.linalg.norm(diff, axis=1)

    def nearest_point(self, pos: np.ndarray): #Find the nearest point on the track to a given position
    diffs = self.waypoints - pos
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return self.waypoints[idx], dists[idx], idx

    def heading_at(self, idx: int) -> float: #Calculate the heading (direction) of the track at a given waypoint index
    nxt = (idx + 1) % len(self.waypoints)
    delta = self.waypoints[nxt] - self.waypoints[idx]
    return float(np.arctan2(delta[1], delta[0]))

    def progress(self, idx: int) -> float:  #Calculate the progress along the track as a fraction of the total length, given a waypoint index
    return float(self.segment_lengths[:idx].sum() / self.total_length)