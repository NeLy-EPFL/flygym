import cv2
import numpy as np

#text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 255, 255)
thickness = 2

def step_game(state, gain_right, gain_left, initiated_legs, sim):
    assert state in ["CPG", "tripod", "single"], "Invalid state"

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

