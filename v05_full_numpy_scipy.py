import cv2
import numpy as np
from scipy.ndimage import convolve

class DisparitySmooth:
    def __init__(self, image_path):
        self.image_data = self.read_image(image_path)

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Image not found or invalid.")
        img = img.astype(np.float32)
        return np.expand_dims(img, axis=(0, 1)) 

    def compute_gradients(self, img):
        sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=np.float32)
        sobel_y = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=np.float32)


        grad_x = convolve(img[0, 0], sobel_x, mode='constant', cval=0)
        grad_y = convolve(img[0, 0], sobel_y, mode='constant', cval=0)


        grad_x = grad_x[1:-1, 1:-1]
        grad_y = grad_y[1:-1, 1:-1]

        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return grad_magnitude[np.newaxis, np.newaxis, :, :] 

    def compute_edge_aware_weights(self, img):
        gradients = self.compute_gradients(img)
        weights = np.exp(-np.abs(gradients))
        return weights

    def compute_differences(self):
        img = self.image_data[:, :, 1:-1, 1:-1]
        weights = self.compute_edge_aware_weights(self.image_data)

        horizontal_diff = np.abs(img[:, :, :, :-1] - img[:, :, :, 1:]) * weights[:, :, :, :-1]
        vertical_diff = np.abs(img[:, :, :-1, :] - img[:, :, 1:, :]) * weights[:, :, :-1, :]
        return horizontal_diff, vertical_diff

    def loss_calc(self):
        horizontal_diff, vertical_diff = self.compute_differences()
        loss = np.sum(horizontal_diff) + np.sum(vertical_diff)
        return loss

if __name__ == "__main__":
    IMG_PATH = 'test_depth_imgs\mi_140.png'
    disparity_model = DisparitySmooth(IMG_PATH)
    loss = disparity_model.loss_calc()
    print("Loss:", loss)
