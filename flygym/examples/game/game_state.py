import threading
import numpy as np
import time
import cv2


class GameState:
    def __init__(self):
        self.lock = threading.Lock()
        self._quit = False
        self._reset = False
        self._state = "CPG"

    def get_quit(self):
        with self.lock:
            return self._quit

    def set_quit(self, value):
        with self.lock:
            self._quit = value

    def get_reset(self):
        with self.lock:
            return self._reset

    def set_reset(self, value):
        with self.lock:
            self._reset = value

    def get_state(self):
        with self.lock:
            return self._state

    def set_state(self, value):
        with self.lock:
            self._state = value


class Game:
    def __init__(self, sim, state, renderer, controls):
        self.sim = sim
        self.state = state
        # So the countdown will be shown right at the beginning of the game loop
        self.state.set_reset(True)
        self.renderer = renderer
        self.controls = controls

        self.possible_states = ["CPG", "tripod", "single"]
        self.start_time = 0
        self.curr_time = 0
        self.crossing_time = 0

        self.prev_gain_left = 0
        self.prev_gain_right = 0

    def step(self, i):
        assert self.state.get_state() in self.possible_states, "Invalid state"

        if self.controls.is_joystick:
            gain_left, gain_right, initiated_legs = self.controls.get_actions(
                self.state.get_state()
            )
        else:
            gain_left, gain_right, initiated_legs = self.controls.get_actions(
                self.state.get_state(), self.prev_gain_left, self.prev_gain_right
            )
            self.prev_gain_left = gain_left
            self.prev_gain_right = gain_right

        if self.state.get_state() == "CPG":
            action = np.array([gain_left, gain_right])
        elif self.state.get_state() == "tripod":
            action = initiated_legs[:2]
        elif self.state.get_state() == "single":
            action = initiated_legs

        obs, _, _, _, _ = self.sim.step(action, self.state.get_state())
        self.renderer.update_speed(obs["forward_vel"], i)

        self.curr_time = time.time()

        return obs

    def render(self):
        img = self.sim.render()[0]
        if not img is None:
            if self.crossed_the_finish_line():
                self.renderer.render_finish_line_image(
                    img.copy(),
                    self.state.get_state(),
                    self.crossing_time,
                    self.curr_time - self.start_time,
                )
            else:
                self.renderer.render_simple_image(
                    img.copy(),
                    self.state.get_state(),
                    self.start_time - self.curr_time,
                )

    def reset(self):
        self.crossing_time = 0
        self.prev_gain_left = 0
        self.prev_gain_right = 0

        self.renderer.reset()
        self.controls.flush_keys()

        obs, info = self.sim.reset()
        img = self.sim.render()[0]
        assert not img is None, "Image is None at reset"
        self.renderer.render_countdown(img.copy(), self.state.get_state(), 3)
        self.state.set_reset(False)
        self.start_time = time.time()

    def crossed_the_finish_line(self):
        if self.sim.flies[0].crossed_finish_line_counter > 0:
            if self.crossing_time <= 0:
                self.crossing_time = self.curr_time - self.start_time
                print(self.crossing_time)
            return True
        return False

    def quit(self):
        self.controls.quit()
        self.renderer.quit()
