import math 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
from tqdm import tqdm
from datetime import datetime 
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy.typing as npt
from numba import jit
import torch


@dataclass
class ImageData:
    Image: torch.Tensor

@dataclass
class EndPoints:
    h_end_points: npt.NDArray
    v_end_points: npt.NDArray

@dataclass
class Diff: 
    HDiff: torch.Tensor
    VDiff: torch.Tensor

@dataclass
class SolvedTerms:
    h_total: float
    v_total: float

class Disparity_Smooth():
    def __init__(self, image_path):
        self.image_data = self.read_image(image_path)
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def read_image(self, path) -> ImageData:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.to("cuda")      
            return ImageData(Image=img_tensor)
        except Exception as e:
            raise RuntimeError(f"Error reading image: {e}")

    def end_point_arrays(self) -> EndPoints:
        img = self.image_data
        H, W = img.Image.shape  

        right_edge_values = img.Image[:, -1]  
        y_coords = torch.arange(H, device='cuda')  
        right_edge_coords = torch.stack((y_coords, torch.full_like(y_coords, W-1)), dim=1)
        right_edge = torch.cat((right_edge_coords.float(), right_edge_values.unsqueeze(1)), dim=1)
        # print(right_edge)


        bottom_edge_values = img.Image[-1, :]  
        x_coords = torch.arange(W, device='cuda') 
        bottom_edge_coords = torch.stack((torch.full_like(x_coords, H-1), x_coords), dim=1)
        bottom_edge = torch.cat((bottom_edge_coords.float(), bottom_edge_values.unsqueeze(0).transpose(0,1)), dim=1)
        # print(bottom_edge)

        return EndPoints(h_end_points=right_edge, v_end_points=bottom_edge)
    
    def difference(self) -> Diff:
        end_points = self.end_point_arrays()

        #Avoid constructing tuples repeatedly
        h_end_points_set = {tuple(point) for point in end_points.h_end_points}
        print(h_end_points_set)

        # h_diff, v_diff = [], []
        # for r in range(self.image_data.Image.shape[1]):
        #     for c in range((self.image_data.Image.shape[0])):
        #         main_pixel = (c, r)
        #         print(main_pixel)
        #         if (main_pixel) in h_end_points_set:
        #             print("present")
        #         else:
        #             print("absent")


if __name__ == "__main__":
    IMG_PATH = "grayscale.png"
    # IMG_PATH = "test_depth_imgs/mi_140.png"
    Disparity_Smooth(IMG_PATH).difference()
