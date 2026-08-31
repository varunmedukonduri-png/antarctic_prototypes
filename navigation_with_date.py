import numpy as np
import heapq
import pickle
import math
import os
from kalman_filter import apply_kalman_to_path

_model_lat = None
_model_lon = None

def _load_models():
    global _model_lat, _model_lon
    if _model_lat is None or _model_lon is None:
        if not os.path.exists('model_lat.pkl') or not os.path.exists('model_lon.pkl'):
            raise FileNotFoundError(
                "Model files (model_lat.pkl, model_lon.pkl) not found. "
                "Please run 'python setup_and_train.py' first."
            )
        _model_lat = pickle.load(open('model_lat.pkl', 'rb'))
        _model_lon = pickle.load(open('model_lon.pkl', 'rb'))
    return _model_lat, _model_lon

def predict_iceberg_path(current_lat, current_lon, wind_dir, wind_speed,
                         current_dir, current_speed, pressure, ice_conc, size,
                         month, day_of_year, year, use_kalman=True):
    model_lat, model_lon = _load_models()
    raw_path = []
    lat, lon = current_lat, current_lon
    for _ in range(12):
        features = np.array([[lat, lon, wind_dir, wind_speed,
                              current_dir, current_speed, pressure, ice_conc, size,
                              month, day_of_year, year]])
        next_lat = model_lat.predict(features)[0]
        next_lon = model_lon.predict(features)[0]
        raw_path.append((round(next_lat, 3), round(next_lon, 3)))
        lat, lon = next_lat, next_lon
    if use_kalman:
        filtered_path = apply_kalman_to_path(raw_path)
        return raw_path, filtered_path
    else:
        return raw_path, raw_path

def a_star_route(start, goal, danger_zones, bounds=(-75, -55, -180, 180)):
    danger_set = set([(round(z[0], 3), round(z[1], 3)) for z in danger_zones])
    def heuristic(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    directions = [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05),
                  (-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05), (0.05, 0.05)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if heuristic(current, goal) < 0.1:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in directions:
            neighbor = (round(current[0] + dx, 3), round(current[1] + dy, 3))
            if not (bounds[0] < neighbor[0] < bounds[1] and bounds[2] < neighbor[1] < bounds[3]):
                continue
            if (round(neighbor[0], 3), round(neighbor[1], 3)) in danger_set:
                continue
            tentative_g = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def hill_climb_optimize(path):
    if len(path) <= 2:
        return path
    optimized = [path[0]]
    for i in range(1, len(path) - 1):
        prev = np.array(optimized[-1])
        curr = np.array(path[i])
        nxt = np.array(path[i+1])
        v1 = curr - prev
        v2 = nxt - curr
        if np.linalg.norm(v1) < 0.001 or np.linalg.norm(v2) < 0.001:
            continue
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cos_angle < -0.9:
            continue
        optimized.append(curr)
    optimized.append(path[-1])
    return optimized

def d_star_lite(start, goal, dynamic_obstacles, bounds=(-75, -55, -180, 180)):
    obs_set = set([(round(z[0], 3), round(z[1], 3)) for z in dynamic_obstacles])
    def heuristic(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    def get_neighbors(node):
        dirs = [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05),
                (-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05), (0.05, 0.05)]
        neighbors = []
        for dx, dy in dirs:
            n = (round(node[0]+dx, 3), round(node[1]+dy, 3))
            if (bounds[0] < n[0] < bounds[1] and bounds[2] < n[1] < bounds[3]):
                if (round(n[0], 3), round(n[1], 3)) not in obs_set:
                    too_close = False
                    for obs in dynamic_obstacles:
                        if heuristic(n, obs) < 0.15:
                            too_close = True
                            break
                    if not too_close:
                        neighbors.append(n)
        return neighbors
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    max_iter = 2000
    iter_count = 0
    while open_set and iter_count < max_iter:
        iter_count += 1
        current = heapq.heappop(open_set)[1]
        if heuristic(current, goal) < 0.08:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return a_star_route(start, goal, dynamic_obstacles)
