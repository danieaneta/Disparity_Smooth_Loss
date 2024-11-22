import torch
import cv2
import torch.nn.functional as F

class Disparity_Smooth():
    def __init__(self, image_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_data = self.read_image(image_path)

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Image not found")
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def compute_gradients(self, img):
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)

        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        return grad_magnitude

    def compute_edge_aware_weights(self, img):
        gradients = self.compute_gradients(img)
        weights = torch.exp(-torch.abs(gradients))
        return weights

    def compute_differences(self):
        img = self.image_data
        weights = self.compute_edge_aware_weights(img)

        horizontal_diff = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]) * weights[:, :, :, :-1]
        vertical_diff = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]) * weights[:, :, :-1, :]
        return horizontal_diff, vertical_diff

    def loss_calc(self):
        horizontal_diff, vertical_diff = self.compute_differences()
        loss = horizontal_diff.sum() + vertical_diff.sum()
        return loss.item()

if __name__ == "__main__":
    # IMG_PATH = 'grayscale.png'
    IMG_PATH = 'test_image_small_02.png'
    disparity_model = Disparity_Smooth(IMG_PATH)
    loss = disparity_model.loss_calc()
    print("Loss:", loss)
