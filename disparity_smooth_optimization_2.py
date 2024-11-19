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
    height: int
    width: int
    total_pixels: int
    image: npt.NDArray
    height_list: list
    width_list: list

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
        self.image_path = image_path
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    def read_image(self) -> ImageData:
        try:
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            height, width, height_list, width_list = img.shape[0], img.shape[1], range(img.shape[0]), range(img.shape[1])
            return ImageData(height=height, width=width, total_pixels=height * width, image=img, height_list=height_list, width_list=width_list)
        except Exception as e:
            raise RuntimeError(f"Error reading image: {e}")
        
    @jit(forceobj=True, looplift=False)
    def end_point_arrays(self) -> EndPoints:
        img_data = self.read_image()
        h_end_ar, v_end_ar = [], []
        
        for c in img_data.height_list:
            end_point = np.array((c, img_data.width_list[-1]))
            h_end_ar.append(end_point)
        h_end_ar_arr = np.empty(shape=len(h_end_ar), dtype=object)
        h_end_ar_arr[:] = h_end_ar

        for r in img_data.width_list:
            end_point = np.array((img_data.height_list[-1], r))
            v_end_ar.append(end_point)
        v_end_ar_arr = np.empty(shape=len(v_end_ar), dtype=object)
        v_end_ar_arr[:] = v_end_ar

        return EndPoints(h_end_points=h_end_ar_arr, v_end_points=v_end_ar_arr)

    def difference(self) -> Diff:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_data = self.read_image()
        image_tensor = torch.from_numpy(img_data.image).float().to(device)
        end_points = self.end_point_arrays()

        h_diff, v_diff = [], []

        for r in tqdm(img_data.width_list):
            for c in img_data.height_list:
                main_pixel_index = (c, r)
                if (main_pixel_index) in [tuple(arr) for arr in end_points.h_end_points]:
                    secondary_pixel_index = (c, r-1)
                else:
                    secondary_pixel_index = (c, r+1)
                diff = abs(float(image_tensor[secondary_pixel_index] - float(image_tensor[main_pixel_index])))
                h_diff.append(diff)
        
        for c in tqdm(img_data.height_list):
            for r in img_data.width_list:
                main_pixel_index = (c, r)
                if (main_pixel_index) in [tuple(arr) for arr in end_points.v_end_points]:
                    secondary_pixel_index = (c-1, r)
                else:
                    secondary_pixel_index = (c+1, r)
                diff = abs(float(image_tensor[secondary_pixel_index] - float(image_tensor[main_pixel_index])))
                v_diff.append(diff)


        h_diff = torch.tensor(h_diff, device=device)
        v_diff = torch.tensor(v_diff, device=device)

        return Diff(HDiff=h_diff, VDiff=v_diff)
    
    @jit(forceobj=True, looplift=True)
    def solving_terms(self) -> SolvedTerms:
        img_data = self.read_image()
        diff = self.difference()

        h_total, v_total = [], []
        for i in range(img_data.total_pixels):
            term_1h, term_1v = diff.HDiff[i], diff.VDiff[i] 
            term_2h, term_2v = (math.e ** (-(abs(diff.HDiff[i])))), (math.e ** (-(abs(diff.VDiff[i]))))
            pixel_totalh = term_1h * term_2h
            pixel_totalv = term_1v * term_2v
            h_total.append(pixel_totalh)
            v_total.append(pixel_totalv)
        
        h_total, v_total = sum(h_total), sum(v_total)
        return SolvedTerms(h_total=h_total, v_total=v_total)

    @jit(forceobj=True, looplift=False)
    def loss_calc(self):
        totals = self.solving_terms()
        loss = totals.h_total + totals.v_total
        return loss

if __name__ == "__main__":
    IMG_PATH = "grayscale.png"
    # # IMG_PATH = "grayscale.png"
    # IMG_PATH = "test_depth_imgs/mi_140.png"
    loss = Disparity_Smooth(IMG_PATH).loss_calc()
    print(loss)
