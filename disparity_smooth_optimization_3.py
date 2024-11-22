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
class TermTotals:
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

        bottom_edge_values = img.Image[-1, :]  
        x_coords = torch.arange(W, device='cuda') 
        bottom_edge_coords = torch.stack((torch.full_like(x_coords, H-1), x_coords), dim=1)
        bottom_edge = torch.cat((bottom_edge_coords.float(), bottom_edge_values.unsqueeze(0).transpose(0,1)), dim=1)

        return EndPoints(h_end_points=right_edge, v_end_points=bottom_edge)
    
    def difference(self) -> Diff:        
        end_points = self.end_point_arrays()

        h_end_points_coords, v_end_points_coords = end_points.h_end_points[:, :2], end_points.v_end_points[:, :2]
        h_end_points_set, v_end_points_set = {tuple(coord.tolist()) for coord in h_end_points_coords}, {tuple(coord.tolist()) for coord in v_end_points_coords}

        h_diff, v_diff = [], []
        for c in tqdm(range(self.image_data.Image.shape[0])):
            for r in range(self.image_data.Image.shape[1]):
                main_pixel = (c, r)
                if main_pixel in h_end_points_set:
                    secondary_pixel_index = (c, (r-1))
                else:
                    secondary_pixel_index = (c, (r+1))
                second_value, main_value = float(self.image_data.Image[secondary_pixel_index]), float(self.image_data.Image[main_pixel])
                difference = np.array(abs(float(second_value - main_value)))
                h_diff.append(difference)
        h_diff_arr = np.empty(shape=len(h_diff), dtype=object)
        h_diff_arr[:] = h_diff

        for r in tqdm(range(self.image_data.Image.shape[1])):
            for c in range(self.image_data.Image.shape[0]):
                main_pixel = (c, r)
                if main_pixel in v_end_points_set:
                    secondary_pixel_index = ((c-1), r)
                else:
                    secondary_pixel_index = ((c+1), r)
                second_value, main_value = float(self.image_data.Image[secondary_pixel_index]), float(self.image_data.Image[main_pixel])
                difference = np.array(abs(float(second_value - main_value)))
                v_diff.append(difference)
        v_diff_arr = np.empty(shape=len(v_diff), dtype=object)
        v_diff_arr[:] = v_diff

        return Diff(HDiff=h_diff_arr, VDiff=v_diff_arr)

    def term_value_totals(self) -> TermTotals:
        diff = self.difference()
        h_total, v_total = [], []

        for i in tqdm(range(self.image_data.Image.shape[0] * self.image_data.Image.shape[1])):
            term_1h, term_1v = diff.HDiff[i], diff.VDiff[i]
            term_2h, term_2v = (math.e ** (-(abs(diff.HDiff[i])))), (math.e ** (-(abs(diff.VDiff[i]))))
            pixel_totalh = term_1h * term_2h
            pixel_totalv = term_1v * term_2v
            h_total.append(pixel_totalh)
            v_total.append(pixel_totalv)
        h_total, v_total = sum(h_total), sum(v_total)
        return TermTotals(h_total=h_total, v_total=v_total)
    
    def loss_calc(self):
        totals = self.term_value_totals()
        loss = totals.h_total + totals.v_total
        return loss

if __name__ == "__main__":
    IMG_PATH = "grayscale.png"
    # IMG_PATH = "test_depth_imgs/mi_140.png"
    # IMG_PATH = 'test_image_small.png'
    loss = Disparity_Smooth(IMG_PATH).loss_calc()
    print(loss)
