import numpy as np
from tqdm import trange
import cv2
import pygame
import threading
import time

from flygym import Camera
from flygym.arena import FlatTerrain, SlalomArena
from flygym.preprogrammed import all_leg_dofs
from flygym.examples.game import TurningController, GameFly

import threading
import numpy as np

from pynput import keyboard
import time
import threading

from utils_game import step_game, prepare_image, crossed_line, put_centered_text

CPG_keys = ['w', 's', 'a', 'd', 'q']
state_keys = ['p', 'o', 'i']
leg_keys = ['t', 'g', 'b', 'z', 'h', 'n']
tripod_keys = ['g', 'h']

# Shared lists to store key presses
pressed_CPG_keys = []
pressed_state_keys = []
pressed_leg_keys = []
pressed_tripod_keys = []

joystick_keys = set()
joystick_axis = np.zeros(2)

lock = threading.Lock()

# Global control variables
_quit = False
_reset = False
_state = "CPG"

# Global variable access functions
def get_quit():
    with lock:
        return _quit

def set_quit(value):
    global _quit
    with lock:
        _quit = value

def get_reset():
    with lock:
        return _reset

def set_reset(value):
    global _reset
    with lock:
        _reset = value

def get_state():
    with lock:
        return _state

def set_state(value):
    global _state
    with lock:
        _state = value

switch_joystick_key = 1
quit_joystick_key =  0

print("Discovering joystick")
joystick_connected = False
# Initialize pygame and joystick
pygame.init()
pygame.joystick.init()
print("Done initializing pygame")

# Check for connected joysticks at the beginning
if pygame.joystick.get_count() > 0:
    print("Joystick connected")
    joystick_connected = True
    print(joystick_connected)

# Define the listener function that will run on a separate thread
def joystick_listener():

    print("done with global")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick {joystick.get_instance_id()} connected and ready for use.")
    n_axes = joystick.get_numaxes() - 1
    n_buttons = joystick.get_numbuttons()
    joystick_buttons_order = [10, 11, 12, 4, 5, 6]

    while True:

        # Check for joystick events
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED and event.instance_id == joystick.get_instance_id():
                print(f"Error: Joystick {joystick.get_instance_id()} disconnected.")
                set_quit(True)
                
                pygame.joystick.quit()
                pygame.quit()
                
                return

        # Poll button states
        for i in range(n_buttons):
            but = joystick.get_button(i)
            if but == switch_joystick_key:
                set_reset(True)
                if get_state() == "CPG":
                    set_state("tripod")
                elif get_state() == "tripod":
                    set_state("single")
                elif get_state() == "single":
                    set_state("CPG")
            elif but == quit_joystick_key:
                set_quit(True)
            elif but in joystick_buttons_order:
                with lock:
                    joystick_keys.add(joystick_buttons_order.index(but))

        # Poll axis states
        for i in range(n_axes):
            axis = joystick.get_axis(i)
            with lock:
                joystick_axis[i] = axis
        
        # Short delay to avoid excessive CPU usage
        #time.sleep(0.01)  # Adjust as needed for responsiveness

def retrieve_joystick_axis():
    with lock:
        pjoystick_axis = joystick_axis.copy()
    return pjoystick_axis
    
def retrieve_joystick_keys():
    with lock:
        pjoystick_keys = joystick_keys.copy()
        joystick_keys.clear()
    return pjoystick_keys

def joystick_control(state):
    initiated_legs = np.zeros(6)
    gain_left = 0
    gain_right = 0

    joystick_keys = retrieve_joystick_keys()
    if state == "single":
        initiated_legs[list(joystick_keys)] = 1
    elif state == "tripod":
        if 2 in joystick_keys: # center button (LH)
            initiated_legs[0] = 1
            initiated_legs[2] = 1
            initiated_legs[4] = 1
        if 3 in joystick_keys:
            initiated_legs[1] = 1
            initiated_legs[3] = 1
            initiated_legs[5] = 1
    elif state == "CPG":        
        ctrl_vector = retrieve_joystick_axis()
        # get axis returns a value between -1 and 1
        # The norm of control vector maps to the norm of [gain_right, gain_left]
        # The axis 1 value, maps to the difference between gain_right and gain_left
        normalized_norm = np.linalg.norm(ctrl_vector)/np.sqrt(2)*1.2
        gain_left = normalized_norm
        gain_right = normalized_norm

        offset_norm = np.abs(ctrl_vector[1]) * (1.2+0.6)
        if ctrl_vector[1] > 0:
            #turn to the right: decrease right gain
            gain_right -= offset_norm
        elif ctrl_vector[1] < 0:
            #turn to the left: decrease left gain
            gain_left -= offset_norm
    else:
        raise ValueError("Invalid state")
    
    return gain_right, gain_left, initiated_legs
    
def on_press(key):
    try:
        key_char = key.char  # Gets the character of the key
    except AttributeError:
        return  # Skips non-character keys like 'shift' or 'ctrl'

    if key_char in CPG_keys:
        pressed_CPG_keys.append(key_char)
    elif key_char in leg_keys:
        pressed_leg_keys.append(key_char)
    elif key_char in tripod_keys:
        pressed_tripod_keys.append(key_char)
    elif key_char == 'i':
        set_reset(True)
        set_state("CPG")
    elif key_char == 'o':
        set_reset(True)
        set_state("tripod")
    elif key_char == 'p':
        set_reset(True)
        set_state("single")
    elif key_char == 'e':  # Stop listener when 'e' is pressed (for example)
        set_quit(True)

def retrieve_keys():
    """Retrieve and clear all recorded key presses."""
    with lock:
        pCPG_keys = pressed_CPG_keys[:]
        pleg_keys = pressed_leg_keys[:]
        ptripod_keys = pressed_tripod_keys[:]
        
        # Clear all lists after retrieval
        pressed_CPG_keys.clear()
        pressed_leg_keys.clear()
        pressed_tripod_keys.clear()

    return pCPG_keys, pleg_keys, ptripod_keys

def sort_keyboard_input(pCPG_keys, pleg_keys, ptripod_keys):
    """Sorts the keys pressed and returns the last one."""
    keys = []
    if pCPG_keys:
        keys.append(max(set(pCPG_keys), key=pCPG_keys.count))
    if pleg_keys:
        keys += list(set(pleg_keys))
    if ptripod_keys:
        keys += list(set(ptripod_keys))

    return keys

def keyboard_control(state, gain_left, gain_right):
    initiated_legs = np.zeros(6)

    # Retrieve all keys pressed since the last call
    keys = sort_keyboard_input(*retrieve_keys())

    for key in keys:
        if key == 'a':
            gain_left = 1.2
            gain_right = 0.4
        elif key == 'd':
            gain_right = 1.2
            gain_left = 0.4
        elif key == 'w':
            gain_right = 1.0
            gain_left = 1.0
        elif key == 's':
            gain_right = -1.0
            gain_left = -1.0
        elif key == 'q':
            gain_left = 0.0
            gain_right = 0.0    
        elif key == 't' and state == "single":
            initiated_legs[0] = 1
        elif key == 'g':
            if state == "single":
                initiated_legs[1] = 1
            elif state == "tripod":
                initiated_legs[0] = 1 
                initiated_legs[2] = 1 
                initiated_legs[4] = 1
        elif key == 'b' and state == "single":
            initiated_legs[2] = 1
        elif key == 'z' and state == "single":
            initiated_legs[3] = 1
        elif key == 'h':
            if state == "single":
                initiated_legs[4] = 1 
            elif state == "tripod":
                initiated_legs[1] = 1
                initiated_legs[3] = 1
                initiated_legs[5] = 1
        elif key == 'n' and state == "single":
            initiated_legs[5] = 1

    return gain_right, gain_left, initiated_legs

gain_left = 0
gain_right = 0
initiated_legs = np.zeros(6)

timestep = 1e-4
actuated_joints = all_leg_dofs
arena = SlalomArena()

contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tarsus3", "Tarsus4", "Tarsus5"]
]

# Configuration initiale
fly = GameFly(init_pose="stretch", actuated_joints=actuated_joints,
          contact_sensor_placements=contact_sensor_placements,
          control="position", self_collisions="none",
          xml_variant="seqik_simple",
          )

print("Done with fly")
window_size = (1280, 720)
cam = Camera(fly=fly, play_speed=0.1, draw_contacts=False, camera_id="Animat/camera_top_zoomout", window_size=window_size, camera_follows_fly_orientation=False, fps=30, play_speed_text=False)
print("Done with cam")

sim = TurningController(
    fly=fly,
    cameras=[cam],
    arena=arena,
    timestep=timestep,
    convergence_coefs=np.ones(6) * 20,
    init_control_mode=get_state()
)

print("Done with sim")

speed_window = 20
speed_list = np.zeros(speed_window)
n = 0

max_slalom_time = 30

if joystick_connected:
    print("Starting joystick listener thread")
    # Start the listener thread
    listener_thread = threading.Thread(target=joystick_listener)
    listener_thread.start()
else:
    print("No joystick connected. Using keyboard control.")
    print("Starting listener thread")
    # Start the keyboard listener thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

print("Starting simulation loop")


start_time = 0
while not get_quit():

    if start_time == 0 or get_reset():
        # Reset the simulation
        obs, info = sim.reset()
        img = sim.render(camera_name='cam', width=1000, height=1000)[0]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        base_img = prepare_image(img, speed_list, get_state(), 0)
        countdown = 3
        while countdown > 0:
            img = put_centered_text(base_img, str(countdown), 10)
            cv2.imshow('BeeBrain', img)
            cv2.waitKey(1)
            time.sleep(1)
            countdown -= 1
        
        img = put_centered_text(base_img, "GO !", 10)
        cv2.imshow('BeeBrain', img)
        cv2.waitKey(1)
        time.sleep(1)
        
        prev_fly_pos = obs['fly'][0][:2]
        # Reset leg properties
        initiated_legs = np.zeros(6)
        gain_left = 0
        gain_right = 0

        set_reset(False)
        start_time = time.time()

    # Update speed list project speed on orientation vector
    fly_or = obs['fly_orientation'][:2]
    norm_fly_or = fly_or / np.linalg.norm(fly_or)
    global_vx = obs["fly"][1][0]
    global_vy = obs["fly"][1][1]
    # Project speed on orientation vector
    speed_list[n % speed_window] = global_vx * norm_fly_or[0] + global_vy * norm_fly_or[1]

    obs = step_game(get_state(), gain_right, gain_left, initiated_legs, sim)

    images = sim.render(camera_name='cam', width=1000, height=1000)
    if not images[0] is None:
        img = cv2.rotate(images[0], cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = prepare_image(img, speed_list, get_state(), time.time()-start_time)
        # Mise à jour de l'image affichée
        cv2.imshow('BeeBrain', img)
        cv2.waitKey(1)

        if joystick_connected:
            gain_right, gain_left, initiated_legs = joystick_control(get_state())
            
        else:
            # the control frequncy should be the same as the rendering frequency
            gain_right, gain_left, initiated_legs = keyboard_control(get_state(), gain_left, gain_right)

        # check finish condition
        fly_pos = obs['fly'][0][:2]
        fly_pos_line = [prev_fly_pos, fly_pos]
        elapsed_time = time.time() - start_time
        if crossed_line(fly_pos_line, sim.arena.finish_line_points) or elapsed_time > max_slalom_time:
            set_reset(True)

            level = 1 if get_state() == "CPG" else 2 if get_state() == "tripod" else 3
            if not elapsed_time > max_slalom_time:
                message = f"Congratulations ! \n You cleared level {level}"
                img = put_centered_text(img, message, 2)
                cv2.imshow('BeeBrain', img)
                cv2.waitKey(1)
                time.sleep(2)
            else:
                if level < 3:
                    message = f"Time's up ! \n Going to level {level+1}"
                else:
                    message = "Time's up ! \n Game over !"
                img = put_centered_text(img, message, 2)
                cv2.imshow('BeeBrain', img)
                cv2.waitKey(1)
                time.sleep(2)

            if get_state() == "CPG":
                set_state("tripod")
            elif get_state() == "tripod":
                set_state("single")
            elif get_state() == "single":
                set_quit(True)

    n+=1

print("Simulation ended")
if joystick_connected:
    # Stop the listener once the loop exits
    listener_thread.join()
else:
    # Stop the listener once the loop exits
    listener.stop()
    listener.join()
# Ferme toutes les fenêtres OpenCV une fois la simulation terminée
cv2.destroyAllWindows()