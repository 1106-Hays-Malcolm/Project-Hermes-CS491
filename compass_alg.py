import math
import state

# --- Macros / constants ---
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
MAX_ANGLE = 25


def distance(px, py, gx, gy):
    dx = gx - px
    dy = gy - py
    return math.sqrt(dx ** 2 + dy ** 2)


def compute_player_azimuth(last_x, last_y, px, py, prev_azimuth=None):
    dx = px - last_x
    dy = py - last_y

    if dx == 0 and dy == 0:
        return prev_azimuth if prev_azimuth is not None else 0.0

    azimuth = math.degrees(math.atan2(dy, dx))
    return (azimuth + 360) % 360


def compute_goal_azimuth(px, py, gx, gy):
    dx = gx - px
    dy = gy - py

    azimuth = math.degrees(math.atan2(dy, dx))
    return (azimuth + 360) % 360


def coef_clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def angle_window(px, py, gx, gy):
    distance_to_goal = distance(px, py, gx, gy)
    map_diagonal = math.sqrt(MAP_WIDTH ** 2 + MAP_HEIGHT ** 2)

    coef = coef_clamp(distance_to_goal / map_diagonal, 0.0, 1.0)

    angle = coef * MAX_ANGLE

    if angle < 2:
        return 1

    return int(round(angle))


def compass_output(last_x, last_y, px, py, gx, gy, prev_azimuth=None):
    player_azimuth = compute_player_azimuth(last_x, last_y, px, py, prev_azimuth)
    objective_azimuth = compute_goal_azimuth(px, py, gx, gy)

    window = angle_window(px, py, gx, gy)

    angle_error = abs((objective_azimuth - player_azimuth + 180) % 360 - 180)
    aligned = angle_error <= window

    prev_azimuth = player_azimuth

    return aligned, objective_azimuth, player_azimuth, prev_azimuth


# Update shared compass state using the latest stored coordinates
def update_compass():
    if (
        state.last_x is None or
        state.last_y is None or
        state.current_x is None or
        state.current_y is None or
        state.objective_x is None or
        state.objective_y is None
    ):
        return

    aligned, obj_az, player_az, prev_az = compass_output(
        state.last_x,
        state.last_y,
        state.current_x,
        state.current_y,
        state.objective_x,
        state.objective_y,
        state.previous_azimuth
    )

    state.aligned = aligned
    state.objective_azimuth = obj_az
    state.player_azimuth = player_az
    state.previous_azimuth = prev_az

    # Use objective azimuth for UI rotation
    state.compass_degrees = obj_az