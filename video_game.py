import numpy as np
from tqdm import trange
import cv2
import pygame

from flygym import Fly, Camera, SingleFlySimulation, get_data_path
from flygym.arena import Tethered, FlatTerrain
from flygym.preprogrammed import all_leg_dofs
from flygym.examples.locomotion import HybridTurningController
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork



#text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

#start pygame
pygame.init()
joysticks = {}


run_time = 1
timestep = 1e-4
target_num_steps = int(run_time / timestep)
actuated_joints = all_leg_dofs
arena = FlatTerrain()

contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]


# Configuration initiale
fly = Fly(init_pose="stretch", actuated_joints=actuated_joints, contact_sensor_placements=contact_sensor_placements, control="position", self_collisions="none", xml_variant="seqik", joint_damping=0.3, joint_stiffness=0.01)
cam = Camera(fly=fly, play_speed=0.1, draw_contacts=False, camera_id="Animat/camera_top", window_size=(1280, 720), camera_follows_fly_orientation=True, fps=30)
sim = HybridTurningController(
    fly=fly,
    cameras=[cam],
    arena=arena,
    timestep=timestep,
)
obs, info = sim.reset()
gain_left = 0
gain_right = 0
action = np.array([gain_right, gain_left])
activated_legs = np.ones(6)
full_control = True
phase_control = False
single_control = False
quit = False

timer_LF = 0
timer_LF_activated = False
timer_LM = 0
timer_LM_activated = False
timer_LH = 0
timer_LH_activated = False
timer_RF = 0
timer_RF_activated = False
timer_RM = 0
timer_RM_activated = False
timer_RH = 0
timer_RH_activated = False
timer_L = 0
timer_L_activated = False
timer_R = 0
timer_R_activated = False


state = 'CPG Control'
waiting = True

x_start = obs['fly'][0][0]
y_start = obs['fly'][0][1]

x_position_0 = x_start
y_position_0 = y_start
count_speed = 0

joystick_connected = False


while(not quit):
    for i in trange(target_num_steps):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True  # Flag that we are done so we exit this loop.

            # Handle hotplugging
            if event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for every
                # joystick, filling up the list without needing to create them manually.
                joy = pygame.joystick.Joystick(event.device_index)
                joysticks[joy.get_instance_id()] = joy
                print(f"Joystick {joy.get_instance_id()} connected")
                joystick_connected = True
                timer_LF = 0
                timer_LF_activated = False
                timer_LM = 0
                timer_LM_activated = False
                timer_LH = 0
                timer_LH_activated = False
                timer_RF = 0
                timer_RF_activated = False
                timer_RM = 0
                timer_RM_activated = False
                timer_RH = 0
                timer_RH_activated = False
                timer_L = 0
                timer_L_activated = False
                timer_R = 0
                timer_R_activated = False

            if event.type == pygame.JOYDEVICEREMOVED:
                del joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")
                joystick_connected = False
                timer_LF = 0
                timer_LF_activated = False
                timer_LM = 0
                timer_LM_activated = False
                timer_LH = 0
                timer_LH_activated = False
                timer_RF = 0
                timer_RF_activated = False
                timer_RM = 0
                timer_RM_activated = False
                timer_RH = 0
                timer_RH_activated = False
                timer_L = 0
                timer_L_activated = False
                timer_R = 0
                timer_R_activated = False

        # For each joystick:
        if joystick_connected:
            for joystick in joysticks.values():

                axes = joystick.get_numaxes()

                axis = np.zeros(axes)

                for i in range(axes):
                    axis[i] = joystick.get_axis(i)

                buttons = joystick.get_numbuttons()
                button = np.zeros(buttons)

                for i in range(buttons):
                    button[i] = joystick.get_button(i)


        x_position_1 = obs['fly'][0][0]
        y_position_1 = obs['fly'][0][1]

        distance = int(np.sqrt((x_position_1 - x_start)**2 + (y_position_1 - y_start)**2))

        if gain_left == 0 and gain_right == 0:
            speed = 0
        else:
            if count_speed == 0:
                speed = np.sqrt(obs['fly'][1][0]**2 + obs['fly'][1][1]**2)
                count_speed += 1
            elif count_speed == 50:
                count_speed = 0
            else:
                count_speed += 1

        # Compute 

        if timer_LF_activated:
            if timer_LF <= 1000:
                timer_LF += 1
            elif timer_LF_activated and timer_LF > 100:
                    timer_LF = 0
                    timer_LF_activated = False
                    activated_legs[0] = 0


        if timer_LM_activated:
            if timer_LM <= 1000:
                timer_LM += 1
            elif timer_LM_activated and timer_LM > 1000:
                timer_LM = 0
                timer_LM_activated = False
                activated_legs[1] = 0
            
        if timer_L_activated:
            if timer_L <= 1000:
                timer_L += 1
            elif timer_L_activated and timer_L > 1000:
                timer_L = 0
                timer_L_activated = False
                activated_legs[0] = 0 
                activated_legs[2] = 0 
                activated_legs[4] = 0


        if timer_LH_activated:
            if timer_LH <= 1000:
                timer_LH += 1
            elif timer_LH_activated and timer_LH > 1000:
                    timer_LH = 0
                    timer_LH_activated = False
                    activated_legs[2] = 0


        if timer_RF_activated:
            if timer_RF <= 1000:
                timer_RF += 1
            elif timer_RF_activated and timer_RF > 1000:
                timer_RF = 0
                timer_RF_activated = False
                activated_legs[3] = 0

                
        if timer_RM_activated:
            if timer_RM <= 1000:
                timer_RM += 1
            elif timer_RM_activated and timer_RM > 1000:
                timer_RM = 0
                timer_RM_activated = False
                activated_legs[4] = 0

        
        if timer_R_activated:
            if timer_R <= 1000:
                timer_R += 1
            elif timer_R_activated and timer_R > 1000:
                timer_R = 0
                timer_R_activated = False
                activated_legs[1] = 0
                activated_legs[3] = 0
                activated_legs[5] = 0


        if timer_RH_activated:
            if timer_RH <= 1000:
                timer_RH += 1
            elif timer_RH_activated and timer_RH > 1000:
                timer_RH = 0
                timer_RH_activated = False
                activated_legs[5] = 0
        

        # Met à jour avec les nouveaux gains
        action = np.array([gain_right, gain_left])

        obs, reward, terminated, truncated, info = sim.step(action, activated_legs)
        
        # Rendu de l'image
        image = sim.render(camera_name='cam', width=1000, height=1000)  # Rendu via la caméra
        
        # Vérifie si l'image est correcte
        if not isinstance(image, list) or len(image) == 0 or image[0] is None:
            continue  # Ignore cette itération si aucune image n'est retournée

        image = image[0]  # Prend le premier élément de la liste
        
        # Assure que c'est un tableau NumPy
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:  # Convertit au format uint8 si pas le cas
                image = image.astype(np.uint8)
        else:
            continue  # Ignore si ce n'est pas un tableau NumPy
        
        image = np.squeeze(image)  # Supprime les dimensions de taille 1
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_txt = cv2.putText(im_rgb, state, org, font, fontScale, color, thickness, cv2.LINE_AA)
        im_distance = cv2.putText(im_txt, f'Distance : {distance} mm', (50, 100), font, fontScale, color, thickness, cv2.LINE_AA)
        im_speed = cv2.putText(im_distance, f'speed : {speed:.1f} mm/s', (50, 150), font, fontScale, color, thickness, cv2.LINE_AA)
        
        # Mise à jour de l'image affichée
        cv2.imshow('Simulation', im_speed)

        if joystick_connected:
            if button[0] == 0:
                waiting = True

            if waiting:
                if button[0] == 1:
                    waiting = False
                    if full_control:
                        full_control = False
                        phase_control = True
                        state = 'Tripod'
                        single_control = False
                        activated_legs = np.zeros(6)
                        action = np.array([1, 1])
                            

                    elif phase_control:
                        phase_control = False
                        full_control = False
                        state = 'Single Leg Control'
                        single_control = True
                        activated_legs = np.zeros(6)
                        action = np.array([1, 1])

                    elif single_control:
                        single_control = False
                        phase_control = False
                        full_control = True
                        state = 'CPG Control'
                        activated_legs = np.ones(6)

                    timer_LF = 0
                    timer_LF_activated = False
                    timer_LM = 0
                    timer_LM_activated = False
                    timer_LH = 0
                    timer_LH_activated = False
                    timer_RF = 0
                    timer_RF_activated = False
                    timer_RM = 0
                    timer_RM_activated = False
                    timer_RH = 0
                    timer_RH_activated = False
                    timer_L = 0
                    timer_L_activated = False
                    timer_R = 0
                    timer_R_activated = False

            if button[1] == 1:
                quit = True
                break


            if full_control:
                if axis[0] > 0.2 and (axis[1] > 0.2 or axis[1] < -0.2):
                    gain_left = -axis[1] * (1- axis[0])
                    gain_right = -axis[1] * axis[0]
                elif axis[0] < -0.2 and (axis[1] > 0.2 or axis[1] < -0.2):
                    gain_left = -axis[1] * -axis[0]
                    gain_right = -axis[1] * (1 + axis[0])
                elif axis[0] > 0.2 and (axis[1] > -0.2 or axis[1] < 0.2):
                    gain_left = axis[0]
                    gain_right = 0
                elif axis[0] < -0.2 and (axis[1] > -0.2 or axis[1] < 0.2):
                    gain_left = 0
                    gain_right = -axis[0]
                elif axis[1] < 0.1 and axis[1] > -0.1 and axis[0] < 0.1 and axis[0] > -0.1:
                    gain_left = 0
                    gain_right = 0
                else:
                    gain_left = -axis[1]
                    gain_right = -axis[1]

            if phase_control:
                gain_left = 1
                gain_right = 1
                if not timer_L_activated:
                    if button[2] == 1:
                        timer_L_activated = True
                        activated_legs[0] = 1 
                        activated_legs[2] = 1 
                        activated_legs[4] = 1
                if not timer_R_activated:
                    if button[3] == 1:
                        timer_R_activated = True
                        activated_legs[1] = 1
                        activated_legs[3] = 1
                        activated_legs[5] = 1

            if single_control:
                gain_left = 1
                gain_right = 1
                if not timer_LF_activated:
                    if button[10] == 1:
                        timer_LF_activated = True
                        activated_legs[0] = 1
                if not timer_LM_activated:
                    if button[11] == 1:
                        timer_LM_activated = True
                        activated_legs[1] = 1
                if not timer_LH_activated:
                    if button[12] == 1:
                        timer_LH_activated = True
                        activated_legs[2] = 1
                if not timer_RF_activated:
                    if button[4] == 1:
                        timer_RF_activated = True
                        activated_legs[3] = 1
                if not timer_RM_activated:
                    if button[5] == 1:
                        timer_RM_activated = True
                        activated_legs[4] = 1
                if not timer_RH_activated:
                    if button[6] == 1:
                        timer_RH_activated = True
                        activated_legs[5] = 1
        else:
            
            # Vérifie si une touche est appuyée
            key = cv2.waitKey(1) & 0xFF

            # Si 'Q' est pressé, augmente le gain gauche et diminue le gain droit (tourner à gauche)
            if key == ord('q'):
                gain_left = 1 
                gain_right = -1

            # Si 'D' est pressé, augmente le gain droit et diminue le gain gauche (tourner à droite)
            if key == ord('d'):
                gain_right = 1 
                gain_left = -1

            # Si 'Z' est pressé, augmente le gain droit et le gain gauche (tout droit)
            if key == ord('z'):
                gain_right = 1 
                gain_left = 1
            
            # Si 'S' est pressé, diminue le gain droit et le gain gauche (recule)
            if key == ord('s'):
                gain_right = -1 
                gain_left = -1

            # Quitte si la touche 'e' est pressée
            if key == ord('e'):
                quit = True
                break
            
            # Si 'A' est pressé, met le gain droit et le gain gauche à 0 (stop)
            if key == ord('a'):
                gain_right = 0 
                gain_left = 0
                
            if key == ord('t'):
                if single_control:
                    if not timer_LF_activated:
                        timer_LF_activated = True
                        activated_legs[0] = 1
                    

            if key == ord('g'):
                if single_control:
                    if not timer_LM_activated:
                        timer_LM_activated = True
                        activated_legs[1] = 1
                elif phase_control:
                    if not timer_L_activated:
                        timer_L_activated = True
                        activated_legs[0] = 1 
                        activated_legs[2] = 1 
                        activated_legs[4] = 1
            
            if key == ord('b'):
                if single_control:
                    if not timer_LH_activated :
                        timer_LH_activated = True
                        activated_legs[2] = 1
                    

            if key == ord('y'):
                if single_control:
                    if not timer_RF_activated:
                        timer_RF_activated = True
                        activated_legs[3] = 1

            if key == ord('h'):
                if single_control:
                    if not timer_RM_activated:
                        timer_RM_activated = True
                        activated_legs[4] = 1 
                elif phase_control:
                    if not timer_R_activated:
                        timer_R_activated = True
                        activated_legs[1] = 1
                        activated_legs[3] = 1
                        activated_legs[5] = 1

            if key == ord('n'):
                if single_control:
                    if not timer_RH_activated:
                        timer_RH_activated = True
                        activated_legs[5] = 1
            
            if key == ord('p'): #tripod control
                phase_control = True
                state = 'Tripod'
                activated_legs = np.zeros(6)
                action = np.array([1, 1])
                full_control = False
                single_control = False
                timer_LF = 0
                timer_LF_activated = False
                timer_LM = 0
                timer_LM_activated = False
                timer_LH = 0
                timer_LH_activated = False
                timer_RF = 0
                timer_RF_activated = False
                timer_RM = 0
                timer_RM_activated = False
                timer_RH = 0
                timer_RH_activated = False
                timer_L = 0
                timer_L_activated = False
                timer_R = 0
                timer_R_activated = False
            
            if key == ord('o'): #single leg control
                single_control = True
                state = 'Single Leg Control'
                activated_legs = np.zeros(6)
                action = np.array([1, 1])
                phase_control = False
                full_control = False
                timer_LF = 0
                timer_LF_activated = False
                timer_LM = 0
                timer_LM_activated = False
                timer_LH = 0
                timer_LH_activated = False
                timer_RF = 0
                timer_RF_activated = False
                timer_RM = 0
                timer_RM_activated = False
                timer_RH = 0
                timer_RH_activated = False
                timer_L = 0
                timer_L_activated = False
                timer_R = 0
                timer_R_activated = False

            if key == ord('i'): #full control
                full_control = True
                state = 'CPG Control'
                activated_legs = np.ones(6)
                phase_control = False
                single_control = False
                timer_LF = 0
                timer_LF_activated = False
                timer_LM = 0
                timer_LM_activated = False
                timer_LH = 0
                timer_LH_activated = False
                timer_RF = 0
                timer_RF_activated = False
                timer_RM = 0
                timer_RM_activated = False
                timer_RH = 0
                timer_RH_activated = False
                timer_L = 0
                timer_L_activated = False
                timer_R = 0
                timer_R_activated = False
        

        

# Ferme toutes les fenêtres OpenCV une fois la simulation terminée
cv2.destroyAllWindows()
