from pathlib import Path
import numpy as np
import math
import cv2

texture_folder = Path("/Users/stimpfli/Desktop/flygym/flygym/data/texture_files")

# Generate texture files
# build 6 images generating the texture for the abdomen we would like to have
# two gradients along the vertical and horizontal axis the sides of the boxes
# will be of multicolored while the bottom top front and back will
# have a single color

texture_name = "Thorax"

anteropost_gradient_colors = [[0, 0, 0], [1, 1, 1]]
ventrodorsal_gradient_colors = [[1, 0, 0], [0, 1, 0]]


# Lets define a function between zeros and one defining the grdient along one axis
def sigmoid_fun(x, shift=0.5, steep=1):
    # Given a x location this function returns the amount of color 1 and color 2
    # that should be mixed to obtain the gradient
    # x is a number between 0 and 1
    # the output is a list of two numbers between 0 and 1 whose sum is 1

    c1 = 1 / (1 + math.exp(-x * steep + shift))
    c2 = 1 - c1

    return np.array([c1, c2])


def piecewise_linear(x, a, b):
    if x < a:
        return np.array([0, 1])
    elif x > b:
        return np.array([1, 0])
    else:
        c1 = (x - a) / (b - a)
        c2 = 1 - c1
        return np.array([c1, c2])


def threshold(x, a):
    if x < a:
        return np.array([0, 1])
    else:
        return np.array([1, 0])


ventrodorsal_threshold = 0.1
ventrodorsal_gradient_fun = threshold

anteropost_threshold = 0.5
anteropost_gradient_fun = threshold

# Lets define the size of the texture
texture_size = 256

cube_anterior = np.ones((texture_size, texture_size, 3)) * anteropost_gradient_colors[0]
cube_posterior = (
    np.ones((texture_size, texture_size, 3)) * anteropost_gradient_colors[1]
)
cube_dorsal = np.ones((texture_size, texture_size, 3)) * ventrodorsal_gradient_colors[0]
cube_ventral = (
    np.ones((texture_size, texture_size, 3)) * ventrodorsal_gradient_colors[1]
)

cube_sides = np.zeros((texture_size, texture_size, 3))

for i in range(texture_size):
    for j in range(texture_size):
        x = i / texture_size
        y = j / texture_size

        anteropost_coef = anteropost_gradient_fun(x, anteropost_threshold)
        ventrodorsal_coef = ventrodorsal_gradient_fun(y, ventrodorsal_threshold)

        anteropost_color = np.sum(
            np.dot(np.expand_dims(anteropost_coef, 0), anteropost_gradient_colors),
            axis=0,
        )
        ventrodorsal_color = np.sum(
            np.dot(np.expand_dims(ventrodorsal_coef, 0), ventrodorsal_gradient_colors),
            axis=0,
        )

        cube_sides[i, j, :] = np.mean([anteropost_color, ventrodorsal_color], axis=0)

        if (
            (i < 1 and j < 1)
            or (i < 1 and j >= texture_size - 1)
            or (i >= texture_size - 1 and j < 1)
            or (i >= texture_size - 1 and j >= texture_size - 1)
        ):
            print(i, j, x, y)
            print(ventrodorsal_color, anteropost_color, cube_sides[i, j, :])

# save each of the images in a new folder named after the texturename
texture_subfolder = texture_folder / texture_name
texture_subfolder.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(texture_subfolder / "cube_anterior.png"), cube_anterior * 255)
cv2.imwrite(str(texture_subfolder / "cube_posterior.png"), cube_posterior * 255)
cv2.imwrite(str(texture_subfolder / "cube_dorsal.png"), cube_dorsal * 255)
cv2.imwrite(str(texture_subfolder / "cube_ventral.png"), cube_ventral * 255)
cv2.imwrite(str(texture_subfolder / "cube_sides.png"), cube_sides * 255)
