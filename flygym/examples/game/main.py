import threading
import time
import cv2
import numpy as np
from game_state import GameState, Game
from controls import JoystickControl, KeyboardControl
from renderer import Renderer
import pygame

from flygym import Camera
from flygym.arena import FlatTerrain, SlalomArena
from flygym.preprogrammed import all_leg_dofs
from flygym.examples.game import TurningController, GameFly

# Initial settings
timestep = 1e-4

contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tarsus3", "Tarsus4", "Tarsus5"]
]

fly_has_crossed_line = False


def main():
    # Game state and control setup
    game_state = GameState()
    renderer = Renderer("BeBrain")
    # Configuration initiale
    fly = GameFly(
        init_pose="stretch",
        actuated_joints=all_leg_dofs,
        contact_sensor_placements=contact_sensor_placements,
        control="position",
        self_collisions="none",
        xml_variant="seqik_simple",
        enable_adhesion = True,
        #draw_adhesion=True,
        actuator_gain=40.0,
        adhesion_force = 5.0,

    )
    cam = Camera(
        fly=fly,
        play_speed=0.1,
        draw_contacts=False,
        camera_id="Animat/camera_back_track",
        window_size=renderer.window_size,
        camera_follows_fly_orientation=False,
        fps=35,
        play_speed_text=False,
    )

    freq = 50
    sim = TurningController(
        fly=fly,
        cameras=[cam],
        arena=SlalomArena(n_gates=1),
        timestep=1e-4,
        convergence_coefs=np.ones(6) * 100,
        intrinsic_freqs= np.ones(6) * freq,
        leg_step_time=1.0 / freq,
        init_control_mode=game_state.get_state(),
    )

    # check if a joystick is connected
    # Initialize pygame and joystick
    pygame.init()
    pygame.joystick.init()

    # Check for connected joysticks at the beginning
    if pygame.joystick.get_count() > 0:
        controls = JoystickControl(game_state)
    else:
        controls = KeyboardControl(game_state)

    s = time.time()
    game = Game(sim, game_state, renderer, controls)
    print("Initialization time: ", time.time() - s)
    n = 0
    while not game.state.get_quit():
        if game.state.get_reset():
            game.reset()

        game.step(n)
        game.render()
        n += 1

    game.quit()


if __name__ == "__main__":
    main()
