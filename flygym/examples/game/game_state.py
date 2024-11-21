import threading
import numpy as np
import time
import pickle


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
        self.state_floor_colors = {"CPG": (0, 0, 1, 1), "tripod": (0, 1, 0, 1), "single": (1, 0, 0, 1)}
        self.start_time = 0
        self.curr_time = 0
        self.crossing_time = 0
        self.sim_crossing_time = 0
        self.start_sim_time = 0

        self.prev_gain_left = 0
        self.prev_gain_right = 0

        # leaderboars is a dictionnary with the mode as key and the list of time the 5 best times
        # check if file is empty
        try:
            with open("leaderboard.pkl", "rb") as f:
                self.all_leaderboards = pickle.load(f)
        except EOFError:
            self.all_leaderboards = {}
        except FileNotFoundError:
            self.all_leaderboards = {}

        self.control_mode = "joystick" if self.controls.is_joystick else "keyboard"
        self.curr_mode = "{}_{}".format(self.state.get_state(), self.control_mode)
        if self.curr_mode not in self.all_leaderboards:
            self.all_leaderboards[self.curr_mode] = []
        else:
            self.curr_leaderboard = self.all_leaderboards[self.curr_mode]

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
        self.sim.cameras[0]._update_cam_pose_follow_fly(self.sim.physics)
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
                    self.curr_leaderboard,
                    self.sim.fly.tot_energy_consumed
                )
            else:
                self.renderer.render_simple_image(
                    img.copy(),
                    self.state.get_state(),
                    self.curr_time - self.start_time,
                    self.curr_leaderboard,
                    self.sim.fly.tot_energy_consumed
                )

    def reset(self):
        
        self.update_leaderboard()

        self.crossing_time = 0
        self.sim_crossing_time = 0
        self.prev_gain_left = 0
        self.prev_gain_right = 0

        self.renderer.reset()
        self.controls.flush_keys()
        self.curr_mode = "{}_{}".format(self.state.get_state(), self.control_mode)
        # load the new leaderboard
        if self.curr_mode not in self.all_leaderboards:
            self.curr_leaderboard = []
        else:
            self.curr_leaderboard = self.all_leaderboards[self.curr_mode]

        self.sim.physics.named.model.geom_rgba["ground"] = self.state_floor_colors[self.state.get_state()]
        obs, info = self.sim.reset()
        img = self.sim.render()[0]
        assert not img is None, "Image is None at reset"
        self.renderer.render_countdown(img.copy(), self.state.get_state(), 3, self.curr_leaderboard)
        self.state.set_reset(False)
        self.start_time = time.time()
        self.start_sim_time = self.sim.curr_time

    def crossed_the_finish_line(self):
        if self.sim.flies[0].crossed_finish_line_counter > 0:
            if self.crossing_time <= 0:
                self.crossing_time = self.curr_time - self.start_time
                self.sim_crossing_time = self.sim.curr_time - self.start_sim_time
                print(self.crossing_time)
            return True
        return False
    
    def update_leaderboard(self):
        # Update the leaderboard
        if self.crossing_time > 0:
            if len(self.curr_leaderboard) <= 0:
                self.curr_leaderboard.append(self.crossing_time)
            elif self.crossing_time < self.curr_leaderboard[-1] or len(self.curr_leaderboard) < 5:
                self.curr_leaderboard.append(self.crossing_time)
                self.curr_leaderboard.sort() # sort based on simulation time
                if len(self.curr_leaderboard) > 5:
                    self.curr_leaderboard = self.curr_leaderboard[:5]
            self.all_leaderboards[self.curr_mode] = self.curr_leaderboard

    def quit(self):
        self.update_leaderboard()
        # save the leaderboard
        with open("leaderboard.pkl", "wb") as f:
            pickle.dump(self.all_leaderboards, f)
        self.controls.quit()
        self.renderer.quit()
