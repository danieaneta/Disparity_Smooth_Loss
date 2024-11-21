import numpy as np
import cv2

image = np.array([
    [0, 85, 170],     # Row 1 with increasing intensity
    [170, 85, 0],     # Row 2 with decreasing intensity
    [255, 127, 63]    # Row 3 with mixed intensity values
], dtype=np.uint8)


cv2.imwrite("grayscale.png", image)