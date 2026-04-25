import time
from screenshot import read_coordinates_from_screen
import state

SCREENSHOT_TIMER = 1

##Logic to control when the vision model is running to keep language models running sequential instead of parallel
def run_vl_loop(model, processor, device, run_flag):
    while True:
        if run_flag["run"]:
            x_coord, y_coord = read_coordinates_from_screen(model, processor, device)

            # Update shared state with new coordinates
            if x_coord != "" and y_coord != "":
                state.last_x = state.current_x
                state.last_y = state.current_y

                state.current_x = int(x_coord)
                state.current_y = int(y_coord)

                print(state.current_x, state.current_y)

        time.sleep(SCREENSHOT_TIMER)