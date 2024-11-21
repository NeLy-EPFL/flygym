import cv2
import time
import numpy as np


class Renderer:
    def __init__(self, window_name, window_size=(1280, 720), speed_window_size=20):
        self.window_name = window_name
        self.window_size = window_size
        self.speed_window_size = speed_window_size
        self.speed_list = np.zeros(speed_window_size)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 255, 255)
        self.base_thickness = 2
        self.text_map = {
            "CPG": "CPG (High level) control",
            "tripod": "Tripod (Mid level) control",
            "single": "Single leg (Low level) control",
        }

        self.leaderboard_boundaries = [(1280-400, 720 - 230), (1280, 720)]
        
        self.consumed_energy_ys = [50, 80]
        self.energy_bar_width = np.rint(0.4*self.window_size[0]).astype(int) # 40% of the window width
        self.energy_mutlipler = 1e6
        self.energy_unit = "microJoules"
        self.energy_bar_min = 0
        self.energy_bar_max = 10
        self.energy_bar_graduations = 10
        self.energy_bar_graduation_increment = self.energy_bar_max // self.energy_bar_graduations

    def initialize_window(self):
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.moveWindow(self.window_name, 0, 0)

    def update_speed(self, speed, i):
        self.speed_list[i % self.speed_window_size] = speed

    def prepare_simple_image(self, image, state, time, consumed_energy):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image = np.squeeze(image)
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_txt = cv2.putText(
            im_rgb,
            self.text_map[state],
            (50, 720 - 3 * 35),
            self.font,
            self.fontScale,
            self.color,
            self.base_thickness,
            cv2.LINE_AA,
        )
        im_speed = cv2.putText(
            im_txt,
            f"speed : {np.mean(self.speed_list)+0.01:.0f} mm/s",
            (50, 720 - 2 * 35),
            self.font,
            self.fontScale,
            self.color,
            self.base_thickness,
            cv2.LINE_AA,
        )
        im_time = cv2.putText(
            im_speed,
            f"time : {time:.2f} s",
            (50, 720 - 35),
            self.font,
            self.fontScale,
            self.color,
            self.base_thickness,
            cv2.LINE_AA,
        )

        img = self.draw_energy_bar(im_time, consumed_energy)

        return img

    def draw_energy_bar(self, img, consumed_energy):
            
        im_energy = cv2.putText(
            img,
            f"Energy consumed {self.energy_unit}",
            (50, self.consumed_energy_ys[0]-15),
            self.font,
            self.fontScale,
            (0, 0, 0),
            self.base_thickness,
            cv2.LINE_AA,
        )
        # add scale (black empty rectangle with spines and 0, 100, 200, ..., 1000 as text)
        img_energy = cv2.rectangle(
            im_energy,
            (50 - 2, self.consumed_energy_ys[0] - 2),
            (50 + 2 + self.energy_bar_width, self.consumed_energy_ys[1] + 2),
            (0, 0, 0),
            2,
        )
        for i in range(self.energy_bar_graduations+1):
            text = f"{i*self.energy_bar_graduation_increment+self.energy_bar_min}"
            ((fw,fh), baseline) = cv2.getTextSize(text,
                                                fontFace=self.font,
                                                fontScale=0.5,
                                                thickness=2) # empty string is good enough
            img_energy = cv2.putText(
                img_energy,
                text,
                (50 + i * self.energy_bar_width // self.energy_bar_graduations - fw//2, self.consumed_energy_ys[1] + fh+6),
                self.font,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        # add filled gree rectangle with consumed energy
        consumed_energy_rescaled = consumed_energy*self.energy_mutlipler - self.energy_bar_min
        consumed_energy_width = np.rint(self.energy_bar_width * consumed_energy_rescaled  / self.energy_bar_max).astype(int)
        img_energy = cv2.rectangle(
            img_energy,
            (50, self.consumed_energy_ys[0]),
            (50 + consumed_energy_width, self.consumed_energy_ys[1]),
            (0, 255, 0),
            -1,
        )

        if consumed_energy_rescaled > self.energy_bar_max:
            self.energy_bar_min += self.energy_bar_max

        return img_energy

    def render_simple_image(self, image, state, time, leaderboard, consumed_energy):
        img = self.prepare_simple_image(image, state, time, consumed_energy)
        img = self.add_leaderboard(img, leaderboard)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def put_centered_text(self, img, message, font_multiplier):
        textsize = cv2.getTextSize(
            message, self.font, self.fontScale * font_multiplier, self.base_thickness
        )[0]
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2
        return cv2.putText(
            img.copy(),
            message,
            (textX, textY),
            self.font,
            self.fontScale * font_multiplier,
            self.color,
            self.base_thickness * font_multiplier,
            cv2.LINE_AA,
        )

    def render_countdown(self, base_img, state, countdown, leaderboard):
        base_img = self.prepare_simple_image(base_img, state, 0, 0)

        for i in range(countdown, 0, -1):
            img = self.put_centered_text(base_img, str(i), 10)
            img = self.add_leaderboard(img, leaderboard)
            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)  # 1ms just to update the window
            time.sleep(1)  # wait for 1 second
        img = self.put_centered_text(base_img, "GO!", 10)
        img = self.add_leaderboard(img, leaderboard)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)  # 1ms just to update the window
        time.sleep(1)  # wait for 1 second

    def render_finish_line_image(self, image, state, crossing_time, time, leaderboard, consumed_energy):
        img = self.prepare_simple_image(image, state, time, consumed_energy)
        img = self.put_centered_text(
            img, f"You are quite the fly: {crossing_time:.3f}s", 2
        )
        img = self.add_leaderboard(img, leaderboard)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def add_leaderboard(self, base_img, leaderboard):
        # add a grey rectangle with alpha = 0.5 to the bottom right corner
        img = base_img.copy()
        overlay = img.copy()
        alpha = 0.5
        overlay = cv2.rectangle(
            overlay,
            self.leaderboard_boundaries[0],
            self.leaderboard_boundaries[1],
            (192, 192, 192),
            -1,
        )
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        for i, line in enumerate(leaderboard):
            img = cv2.putText(
                img,
                f"{i+1}. {line:.3f}s",
                (self.leaderboard_boundaries[0][0] + 100, self.leaderboard_boundaries[0][1] + 40 + i * 40),
                self.font,
                self.fontScale,
                self.color,
                self.base_thickness,
                cv2.LINE_AA,
            )
        
        return img

    def reset(self):
        self.energy_bar_min = 0
        self.speed_list = np.zeros(self.speed_window_size)

    def quit(self):
        cv2.destroyAllWindows()
