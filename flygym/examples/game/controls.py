import threading
import numpy as np
import pygame
from pynput import keyboard
from game_state import GameState
import time

# abstract class
from abc import ABC, abstractmethod
import threading


class Controls(ABC):
    @abstractmethod
    def __init__(selfgame_state: GameState):
        """Initialize the controls and start the listener thread"""
        pass

    @abstractmethod
    def listener(self):
        """Function run in the listener threads"""
        pass

    @abstractmethod
    def retrieve_keys(self):
        """Retrieve the pressed keys since last call"""
        pass

    @abstractmethod
    def get_action(self):
        """get the keys and translates it into actions (performable by the simulation )"""
        pass

    def quit(self):
        """Quit and cleans control handles threads ect"""
        pass

    def flush_keys(self):
        """Flush the keys"""
        pass


class JoystickControl:
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        pygame.event.get()  # Flush any residual events
        pygame.event.clear()

        print(
            f"Joystick {self.joystick.get_instance_id()} connected and ready for use."
        )

        self.n_axes = self.joystick.get_numaxes() - 1
        self.n_buttons = self.joystick.get_numbuttons()
        self.joystick_buttons_order = [10, 11, 12, 4, 5, 6] # LF, LM, LH, RF, RM, RH
        self.backward_joystick_buttons_order = [15, 14, 13, 9, 8, 7]
        self.is_joystick = True
        time.sleep(0.1)  # leave tie for joystick to be ready ??

        self.joystick_leg_presses = np.zeros(6)
        self.joystick_axis = np.zeros(2)

        self.lock = threading.Lock()

        self.listener_thread = threading.Thread(target=self.listener)
        self.listener_thread.start()

        # Keyboard listener to quit the game
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()

    def on_press(self, key):
        key = key.char if hasattr(key, 'char') else str(key)
        # get escape key to quit
        if key == "Key.esc":
            self.game_state.set_quit(True)
            print("Quitting")
        if key == "Key.space":
            self.game_state.set_reset(True)
            print("Resetting")
        if key == "i":
            self.game_state.set_reset(True)
            self.game_state.set_state("CPG")
        elif key == "o":
            self.game_state.set_reset(True)
            self.game_state.set_state("tripod")
        elif key == "p":
            self.game_state.set_reset(True)
            self.game_state.set_state("single")

    def listener(self):
        buttons = np.zeros(self.n_buttons)

        while self.game_state.get_quit() == False:
            for event in pygame.event.get():
                if (
                    event.type == pygame.JOYDEVICEREMOVED
                    and event.instance_id == self.joystick.get_instance_id()
                ):
                    print(
                        f"Error: Joystick {self.joystick.get_instance_id()} disconnected."
                    )
                    self.game_state.set_quit(True)

            # Poll button states
            for i in range(self.n_buttons):
                buttons[i] = self.joystick.get_button(i)

            for j, pressed in enumerate(buttons[self.joystick_buttons_order]):
                if pressed:
                    with self.lock:
                        self.joystick_leg_presses[j] = 1
            for j, pressed in enumerate(buttons[self.backward_joystick_buttons_order]):
                if pressed:
                    with self.lock:
                        self.joystick_leg_presses[j] = -1

            for i in range(self.n_axes - 1):
                axis = self.joystick.get_axis(i)
                with self.lock:
                    self.joystick_axis[i] = axis

            time.sleep(0.1)  # prevent busy waiting

        pygame.joystick.quit()
        pygame.quit()

    def retrieve_joystick_axis(self):
        with self.lock:
            pjoystick_axis = self.joystick_axis.copy()
        return pjoystick_axis

    def retrieve_joystick_buttons(self):
        with self.lock:
            pjoystick_legs = self.joystick_leg_presses.copy()
            self.joystick_leg_presses = np.zeros(6)
        return pjoystick_legs

    def get_actions(self, state):
        initiated_legs = np.zeros(6)
        gain_left = 0
        gain_right = 0

        joystick_leg_presses = self.retrieve_joystick_buttons()
        if state == "single":
            initiated_legs = joystick_leg_presses
        elif state == "tripod":
            left_tripod_press = joystick_leg_presses[2]
            initiated_legs[0] = left_tripod_press
            initiated_legs[2] = left_tripod_press
            initiated_legs[4] = left_tripod_press
            right_tripod_press = joystick_leg_presses[5]
            initiated_legs[1] = right_tripod_press
            initiated_legs[3] = right_tripod_press
            initiated_legs[5] = right_tripod_press
        elif state == "CPG":
            ctrl_vector = self.retrieve_joystick_axis()
            # get axis returns a value between -1 and 1
            # The norm of control vector maps to the norm of [gain_right, gain_left]
            # The axis 1 value, maps to the difference between gain_right and gain_left
            normalized_norm = np.linalg.norm(ctrl_vector) / np.sqrt(2) * 1.2
            gain_left = normalized_norm * (-1) * np.sign(ctrl_vector[1])
            gain_right = normalized_norm * (-1) * np.sign(ctrl_vector[1])

            offset_norm = np.abs(ctrl_vector[0]) * 0.6
            if ctrl_vector[0] > 0:
                # turn to the right: decrease right gain
                gain_right -= offset_norm
            elif ctrl_vector[0] < 0:
                # turn to the left: decrease left gain
                gain_left -= offset_norm
        else:
            raise ValueError("Invalid state")

        return gain_left, gain_right, initiated_legs

    def flush_keys(self):
        with self.lock:
            self.joystick_leg_presses = np.zeros(6)

    def quit(self):
        if not self.game_state.get_quit():
            self.game_state.set_quit(True)
        self.listener_thread.join()
        self.flush_keys()
        pygame.joystick.quit()
        pygame.quit()
        self.keyboard_listener.stop()
        self.keyboard_listener.join()


class KeyboardControl:
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.is_joystick = False

        # Keyboard keys
        self.CPG_keys = ["w", "s", "a", "d", "q"]
        self.state_keys = ["p", "o", "i"]
        self.leg_keys = ["t", "g", "b", "z", "h", "n", "r", "f", "v", "u", "j", "m"]
        self.tripod_keys = ["g", "h", "f", "j"]

        # Shared lists to store key presses
        self.pressed_CPG_keys = []
        self.pressed_state_keys = []
        self.pressed_leg_keys = []
        self.pressed_tripod_keys = []

        self.lock = threading.Lock()

        print("Starting key listener")
        # Start the keyboard listener thread
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        key_str = key.char if hasattr(key, 'char') else str(key)  # Gets the character of the key

        if key_str in self.CPG_keys:
            self.pressed_CPG_keys.append(key_str)
        elif key_str in self.leg_keys:
            self.pressed_leg_keys.append(key_str)
        elif key_str in self.tripod_keys:
            self.pressed_tripod_keys.append(key_str)
        elif key_str == "i":
            self.game_state.set_reset(True)
            self.game_state.set_state("CPG")
        elif key_str == "o":
            self.game_state.set_reset(True)
            self.game_state.set_state("tripod")
        elif key_str == "p":
            self.game_state.set_reset(True)
            self.game_state.set_state("single")
        elif key_str == "Key.esc":  # Quit when esc is pressed
            self.game_state.set_quit(True)
        elif key_str == "Key.space":
            self.game_state.set_reset(True)

    def retrieve_keys(self):
        """Retrieve and clear all recorded key presses."""
        with self.lock:
            pCPG_keys = self.pressed_CPG_keys[:]
            self.pressed_CPG_keys.clear()
        with self.lock:
            pleg_keys = self.pressed_leg_keys[:]
            self.pressed_leg_keys.clear()
        with self.lock:
            ptripod_keys = self.pressed_tripod_keys[:]
            self.pressed_tripod_keys.clear()

        return pCPG_keys, pleg_keys, ptripod_keys

    def sort_keyboard_input(self, pCPG_keys, pleg_keys, ptripod_keys):
        """Sorts the keys pressed and returns the last one."""
        keys = []
        if pCPG_keys:
            keys.append(max(set(pCPG_keys), key=pCPG_keys.count))
        if pleg_keys:
            keys += list(set(pleg_keys))
        if ptripod_keys:
            keys += list(set(ptripod_keys))

        return keys

    def get_actions(self, state, prev_gain_left, prev_gain_right):
        initiated_legs = np.zeros(6)
        gain_left = prev_gain_left
        gain_right = prev_gain_right

        # Retrieve all keys pressed since the last call
        keys = self.sort_keyboard_input(*self.retrieve_keys())

        for key in keys:
            if key == "a":
                gain_left = 0.4
                gain_right = 1.2
            elif key == "d":
                gain_right = 0.4
                gain_left = 1.2
            elif key == "w":
                gain_right = 1.0
                gain_left = 1.0
            elif key == "s":
                gain_right = -1.0
                gain_left = -1.0
            elif key == "q":
                gain_left = 0.0
                gain_right = 0.0
            elif key == "t" and state == "single":
                initiated_legs[0] = 1
            elif key == "g":
                if state == "single":
                    initiated_legs[1] = 1
                elif state == "tripod":
                    initiated_legs[0] = 1
                    initiated_legs[2] = 1
                    initiated_legs[4] = 1
            elif key == "b" and state == "single":
                initiated_legs[2] = 1
            elif key == "z" and state == "single":
                initiated_legs[3] = 1
            elif key == "h":
                if state == "single":
                    initiated_legs[4] = 1
                elif state == "tripod":
                    initiated_legs[1] = 1
                    initiated_legs[3] = 1
                    initiated_legs[5] = 1
            elif key == "n" and state == "single":
                initiated_legs[5] = 1
            elif key == "r" and state == "single":
                initiated_legs[0] = -1
            elif key == "f":
                if state == "single":
                    initiated_legs[1] = -1
                elif state == "tripod":
                    initiated_legs[0] = -1
                    initiated_legs[2] = -1
                    initiated_legs[4] = -1
            elif key == "v" and state == "single":
                initiated_legs[2] = -1
            elif key == "u" and state == "single":
                initiated_legs[3] = -1
            elif key == "j":
                if state == "single":
                    initiated_legs[4] = -1
                elif state == "tripod":
                    initiated_legs[1] = -1
                    initiated_legs[3] = -1
                    initiated_legs[5] = -1
            elif key == "m" and state == "single":
                initiated_legs[5] = -1
                
        return gain_left, gain_right, initiated_legs

    def flush_keys(self):
        with self.lock:
            self.pressed_CPG_keys.clear()
            self.pressed_leg_keys.clear()
            self.pressed_tripod_keys.clear()

    def quit(self):
        self.listener.stop()
        self.listener.join()
        self.game_state.set_quit(True)
        self.flush_keys()
