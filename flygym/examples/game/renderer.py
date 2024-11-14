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

    def initialize_window(self):
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.moveWindow(self.window_name, 0, 0)

    def update_speed(self, speed, i):
        self.speed_list[i % self.speed_window_size] = speed

    def prepare_simple_image(self, image, state, time):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image = np.squeeze(image)
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_txt = cv2.putText(
            im_rgb,
            self.text_map[state],
            (50, 720 - 3 * 30),
            self.font,
            self.fontScale,
            self.color,
            self.base_thickness,
            cv2.LINE_AA,
        )
        im_speed = cv2.putText(
            im_txt,
            f"speed : {np.mean(self.speed_list)+0.01:.0f} mm/s",
            (50, 720 - 2 * 30),
            self.font,
            self.fontScale,
            self.color,
            self.base_thickness,
            cv2.LINE_AA,
        )
        im_time = cv2.putText(
            im_speed,
            f"time : {time:.2f} s",
            (50, 720 - 30),
            self.font,
            self.fontScale,
            self.color,
            self.base_thickness,
            cv2.LINE_AA,
        )
        return im_time

    def render_simple_image(self, image, state, time):
        img = self.prepare_simple_image(image, state, time)
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

    def render_countdown(self, base_img, state, countdown):
        base_img = self.prepare_simple_image(base_img, state, 0)

        for i in range(countdown, 0, -1):
            img = self.put_centered_text(base_img, str(i), 10)
            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)  # 1ms just to update the window
            time.sleep(1)  # wait for 1 second
        img = self.put_centered_text(base_img, "GO!", 10)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)  # 1ms just to update the window
        time.sleep(1)  # wait for 1 second

    def render_finish_line_image(self, image, state, crossing_time, time):
        img = self.prepare_simple_image(image, state, time)
        img = self.put_centered_text(
            img, f"You are quite the fly: {crossing_time:.3f} s", 2
        )
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def reset(self):
        self.speed_list = np.zeros(self.speed_window_size)

    def quit(self):
        cv2.destroyAllWindows()