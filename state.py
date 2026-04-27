# state.py does not run anything by itself it only holds values for updates as needed

# Vision / coordinate state
last_x = None
last_y = None

current_x = None
current_y = None

# Objective coordinate state
objective_x = None
objective_y = None

# Compass state
previous_azimuth = None
player_azimuth = 0.0
objective_azimuth = 0.0
aligned = False

# Vision loop control
vision_running = False