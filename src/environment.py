import numpy as np
import gymnasium as gym
from gymnasium import spaces
from track import Track
from config import (
    N_POINTS, SPEED, STEER_ANGLE, MAX_STEPS,
    TRACK_HALF_WIDTH, DIST_PENALTY_K, HEADING_BONUS_K,
    OFF_TRACK_REWARD, STEP_SURVIVAL
)

def __init__(self, x, y):
    super().__init__()
    self.track = Track(x, y, N_POINTS)

    obs_low  = np.array([-10, -10, -1, -1, 0, -5, -np.pi], dtype=np.float32)
    obs_high = np.array([ 10,  10,  1,  1, 1,  5,  np.pi], dtype=np.float32)
    self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
    self.action_space      = spaces.Discrete(3)
    self._pos     = np.zeros(2, dtype=np.float64)
    self._heading = 0.0
    self._step    = 0
    self._wp_idx  = 0
    
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    
    n = len(self.track.waypoints)
    start_idx = self.np_random.integers(0, n)
    self._pos     = self.track.waypoints[start_idx].copy()
    self._heading = self.track.heading_at(start_idx)
    self._step    = 0
    self._wp_idx  = start_idx
    
    self._pos += self.np_random.uniform(-0.05, 0.05, size=2)    #Adds a tiny random nudge to avoid overfitting.
    
    return self._get_state(), {}
    
def step(self, action: int):
    delta = {0: -STEER_ANGLE, 1: 0.0, 2: STEER_ANGLE}[action]
    self._heading += delta
    self._heading  = (self._heading + np.pi) % (2 * np.pi) - np.pi

    self._pos[0] += SPEED * np.cos(self._heading)
    self._pos[1] += SPEED * np.sin(self._heading)
    self._step   += 1

    _, dist, self._wp_idx = self.track.nearest_point(self._pos)

    terminated = False
    if dist > TRACK_HALF_WIDTH:
        reward     = OFF_TRACK_REWARD
        terminated = True
    else:
        heading_error = self._heading_error()
        reward = (
            STEP_SURVIVAL
            - DIST_PENALTY_K  * dist
            + HEADING_BONUS_K * np.cos(heading_error)
        )

    truncated = self._step >= MAX_STEPS
    return self._get_state(), float(reward), terminated, truncated, {}

def _get_state(self) -> np.ndarray:
    _, dist, self._wp_idx = self.track.nearest_point(self._pos)
    heading_error = self._heading_error()
    return np.array([
        self._pos[0],
        self._pos[1],
        np.cos(self._heading),
        np.sin(self._heading),
        SPEED,
        dist,
        heading_error,
    ], dtype=np.float32)

def _heading_error(self) -> float:
    track_heading = self.track.heading_at(self._wp_idx)
    err = self._heading - track_heading
    return float((err + np.pi) % (2 * np.pi) - np.pi)
