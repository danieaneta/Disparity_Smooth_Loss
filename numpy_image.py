import numpy as np
import cv2

image = np.array([
    [0, 15, 20],     # Row 1 with increasing intensity
    [12, 17, 19],     # Row 2 with decreasing intensity
    [20, 16, 11]    # Row 3 with mixed intensity values
], dtype=np.uint8)


cv2.imwrite("grayscale_updated.png", image)