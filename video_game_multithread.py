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

CPG_keys = ['w', 's', 'a', 'd', 'q']
state_keys = ['p', 'o', 'i']
leg_keys = ['t', 'g', 'b', 'z', 'h', 'n']
tripod_keys = ['g', 'h']

# Shared lists to store key presses
pressed_CPG_keys = []
pressed_state_keys = []
pressed_leg_keys = []
pressed_tripod_keys = []

lock = threading.Lock()
quit = False
reset = False

def on_press(key):
    global quit, reset, state
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
        reset = True
        state = 'CPG'
    elif key_char == 'o':
        reset = True
        state = 'tripod'
    elif key_char == 'p':
        reset = True
        state = 'single'
    elif key_char == 'e':  # Stop listener when 'e' is pressed (for example)
        quit = True

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

def step_game(state, prev_state, gain_right, gain_left, initiated_legs):
    assert state in ["CPG", "tripod", "single"], "Invalid state"

    if state != prev_state:
        if prev_state == "CPG":
            action = np.array([0, 0])
        elif prev_state == "tripod":
            action = np.zeros(2)
        elif prev_state == "single":
            action = np.zeros(6)
        for i in range(100):
            _ = sim.step(action, prev_state)

    if state == "CPG":
        action = np.array([gain_right, gain_left])
    elif state == "tripod":
        action = initiated_legs[:2]
    elif state == "single":
        action = initiated_legs

    obs, _, _, _, _ = sim.step(action, state)


    return obs

def prepare_image(image, speed_list, state, time):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    image = np.squeeze(image)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_txt = cv2.putText(im_rgb, state, (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
    im_speed = cv2.putText(im_txt, f'speed : {np.mean(speed_list)+0.01:.0f} mm/s', (50, 80), font, fontScale, color, thickness, cv2.LINE_AA)
    im_time = cv2.putText(im_speed, f'time : {time:.2f} s', (50, 110), font, fontScale, color, thickness, cv2.LINE_AA)

    return im_time

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

    return state, gain_right, gain_left, initiated_legs

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def crossed_line(fly_pos_line, finish_line):
    # Check if the ply pos line [[x1, y1], [x2, y2]] crossed the finish line [[x1, y1], [x2, y2]]

    return intersect(fly_pos_line[0], fly_pos_line[1], finish_line[0], finish_line[1])

def put_centered_text(img, message, font_multiplier, y_offset = 0):
    textsize = cv2.getTextSize(message, font, fontScale*font_multiplier, thickness)[0]
    if textsize[0] > img.shape[1]:
        # cut the message
        messages = message.split("\n")
        for i, message in enumerate(messages):
            textsize = cv2.getTextSize(message, font, fontScale*font_multiplier, thickness)[0]
            img = put_centered_text(img, message, font_multiplier, y_offset=int(i*textsize[1]+1*font_multiplier))
        return img
    else:
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2
    return cv2.putText(img.copy(), message, (textX, textY+y_offset), font, fontScale*font_multiplier, color, thickness*font_multiplier, cv2.LINE_AA)


#text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 255, 255)
thickness = 2

#start pygame
pygame.init()
joysticks = {}

state = 'CPG'
prev_state = 'CPG'
waiting = True

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
          #joint_damping=0.3, joint_stiffness=0.01
          )

window_size = (1280, 720)
cam = Camera(fly=fly, play_speed=0.1, draw_contacts=False, camera_id="Animat/camera_top_zoomout", window_size=window_size, camera_follows_fly_orientation=False, fps=30, play_speed_text=False)
sim = TurningController(
    fly=fly,
    cameras=[cam],
    arena=arena,
    timestep=timestep,
    convergence_coefs=np.ones(6) * 20,
    init_control_mode=state
)

speed_window = 20
speed_list = np.zeros(speed_window)
n = 0

max_slalom_time = 30

joystick_connected = False

print("Starting listener thread")

# Start the keyboard listener thread
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Starting simulation loop")

start_time = 0
reset = True
while not quit:

    if reset:
        # Reset the simulation
        obs, info = sim.reset()
        img = sim.render(camera_name='cam', width=1000, height=1000)[0]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        base_img = prepare_image(img, speed_list, state, 0)
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

        
        reset = False
        start_time = time.time()

    # Update speed list project speed on orientation vector
    fly_or = obs['fly_orientation'][:2]
    norm_fly_or = fly_or / np.linalg.norm(fly_or)
    global_vx = obs["fly"][1][0]
    global_vy = obs["fly"][1][1]
    # Project speed on orientation vector
    speed_list[n % speed_window] = global_vx * norm_fly_or[0] + global_vy * norm_fly_or[1]

    obs = step_game(state, prev_state, gain_right, gain_left, initiated_legs)
    prev_state = state

    images = sim.render(camera_name='cam', width=1000, height=1000)
    if not images[0] is None:
        img = cv2.rotate(images[0], cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = prepare_image(img, speed_list, state, time.time()-start_time)
        # Mise à jour de l'image affichée
        cv2.imshow('BeeBrain', img)
        cv2.waitKey(1)

        # the control frequncy should be the same as the rendering frequency
        state, gain_right, gain_left, initiated_legs = keyboard_control(state, gain_left, gain_right)

        # Check if the fly crossed the finish line
        fly_pos = obs['fly'][0][:2]
        fly_pos_line = [prev_fly_pos, fly_pos]

        elapsed_time = time.time() - start_time

        if crossed_line(fly_pos_line, sim.arena.finish_line_points) or elapsed_time > max_slalom_time:
            reset = True

            level = 1 if state == "CPG" else 2 if state == "tripod" else 3
            if not elapsed_time > max_slalom_time:
                message = f"Congratulations ! \n You cleared level {level}"
                img = put_centered_text(img, message, 2)
                cv2.imshow('BeeBrain', img)
                cv2.waitKey(1)
                time.sleep(2)
            else:
                message = f"Time's up ! \n Going to level {level+1}"
                img = put_centered_text(img, message, 2)
                cv2.imshow('BeeBrain', img)
                cv2.waitKey(1)
                time.sleep(2)

            if state == "CPG":
                state = "tripod"
            elif state == "tripod":
                state = "single"
            elif state == "single":
                quit = True

    n+=1

print("Simulation ended")
# Stop the listener once the loop exits
listener.stop()
listener.join()
# Ferme toutes les fenêtres OpenCV une fois la simulation terminée
cv2.destroyAllWindows()