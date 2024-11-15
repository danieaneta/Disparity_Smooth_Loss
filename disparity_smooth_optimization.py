import math 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
from tqdm import tqdm
from datetime import datetime 

time = datetime.now()
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")


class Disparity_Smooth():
    def __init__(self, image_path):
        self.image_path = image_path

    def read_image(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        return img, shape
    
    def get_perimeter(self, shape):
        height, width = shape[0], shape[1]
        height_list, width_list = range(height), range(width)
        total_pixels = height * width
        return height, width, height_list, width_list, total_pixels
    
    def h_end_pixel_array(self, width_list, height_list):
        h_end_ar = []
        for c in tqdm(height_list):
            end_point = np.array((c, width_list[-1]))
            h_end_ar.append(end_point)
        h_end_ar_arr = np.empty(shape=len(h_end_ar), dtype=object)
        h_end_ar_arr[:] = h_end_ar
        return h_end_ar_arr

    def v_end_pixel_array(self, width_list, height_list):
        v_end_ar = []
        for r in tqdm(width_list):
            end_point = np.array((height_list[-1], r))
            v_end_ar.append(end_point)
        v_end_ar_arr = np.empty(shape=len(v_end_ar), dtype=object)
        v_end_ar_arr[:] = v_end_ar
        return v_end_ar_arr
    

    def horizontal_diff(self, img, height_list, width_list, h_end_points):
        h_diff = []
        for c in tqdm(height_list):
            for r in width_list:
                main_pixel_index = (c, r)
                # print("STARTING: ", main_pixel_index)
                if any(np.array_equal(main_pixel_index, arr) for arr in h_end_points):
                    secondary_pixel_index = (c, (r-1))
                else: 
                    secondary_pixel_index = (c, (r+1))
                second_pixel, main_pixel = float(img[secondary_pixel_index]), float(img[c, r])
                difference = abs(float(second_pixel - main_pixel))
                diff_np = np.array(difference)
                h_diff.append(diff_np)
        h_diff_arr = np.empty(shape=len(h_diff), dtype=object)
        h_diff_arr[:] = h_diff
        with open(f'logs/diff-logs/h_diff_log_{timestamp}.txt', "w") as file:
            file.write(str(h_diff_arr))

        return h_diff_arr
    
    def vertical_diff(self, img, height_list, width_list, v_end_points):
        v_diff = []
        for r in tqdm(width_list):
            for c in height_list:
                main_pixel_index = (c, r)
                if any(np.array_equal(main_pixel_index, arr) for arr in v_end_points):
                    secondary_pixel_index = ((c-1), r)
                else:
                    secondary_pixel_index = ((c+1), r)
                second_pixel, main_pixel = float(img[secondary_pixel_index]), float(img[c, r])
                difference = abs(float(second_pixel - main_pixel))
                diff_np = np.array(difference)
                v_diff.append(diff_np)
        v_diff_arr = np.empty(shape=len(v_diff), dtype=object)
        v_diff_arr[:] = v_diff

        with open(f'logs/diff-logs/v_diff_log_{timestamp}.txt', "w") as file:
            file.write(str(v_diff_arr))

        return v_diff_arr
    
    def horizontal_term_x(self, h_diff, height_list, h_intensity_diff, total_pixels):
        h_total = []
        for c in tqdm(range(total_pixels)):
            term_1 = h_diff[c]
            term_2 = math.e ** (-(abs(h_intensity_diff[c])))
            h_pixel_diff_logs = f"T1: {term_1}, T2: {term_2}" + "\n"
            h_pixel_pre_exp = f"T1: {term_1}, T2: {h_intensity_diff[c]}" + "\n"
            with open(f"logs/term-logs/h_term_x_logs_{timestamp}.txt", "a") as file:
                file.write(str(h_pixel_diff_logs))
            with open(f"logs/term-logs/h_term_x_logs_pre_exp_{timestamp}.txt", "a") as file:
                file.write(str(h_pixel_pre_exp))
            pixel_total = term_1 * term_2
            h_total.append(pixel_total)
        h_total = sum(h_total)
        print("-------------------------")
        print("H_Term: ", h_total)
        print("-------------------------")
        return h_total
    
    def vertical_term_y(self, v_diff, width_list, v_intensity_diff, total_pixels):
        v_total = []
        for r in tqdm(range(total_pixels)):
            term_1 = v_diff[r]
            term_2 = math.e ** (-(abs(v_intensity_diff[r])))
            v_pixel_diff_logs = f"T1: {term_1}, T2: {term_2}" + "\n"
            v_pixel_pre_exp = f"T1: {term_1}, T2: {v_intensity_diff[r]}" + "\n"
            with open(f"logs/term-logs/v_term_x_logs_{timestamp}.txt", "a") as file:
                file.write(str(v_pixel_diff_logs))
            with open(f"logs/term-logs/v_term_x_logs_pre_exp_{timestamp}.txt", "a") as file:
                file.write(str(v_pixel_pre_exp))
            pixel_total = term_1 * term_2
            v_total.append(pixel_total)
        v_total = sum(v_total)
        print("-------------------------")
        print("V_Term: ", v_total)
        print("-------------------------")
        return v_total
    
    def loss_calc(self, h_total, v_total):
        loss = h_total * v_total
        return loss

    def calculate(self):
        img, shape = self.read_image()
        height, width, height_list, width_list, total_pixels = self.get_perimeter(shape)
        h_end_points = self.h_end_pixel_array(width_list, height_list)
        v_end_points = self.v_end_pixel_array(width_list, height_list)
        h_diff_arr = self.horizontal_diff(img, height_list, width_list, h_end_points)
        v_diff_arr = self.vertical_diff(img, height_list, width_list, v_end_points)
        h_intensity_diff  = h_diff_arr
        v_intensity_diff = v_diff_arr
        h_total = self.horizontal_term_x(h_diff_arr, height_list, h_intensity_diff, total_pixels)
        v_total = self.vertical_term_y(v_diff_arr, width_list, v_intensity_diff, total_pixels)
        loss = self.loss_calc(h_total, v_total)
        print("-------------------------")
        print("LOSS: ", loss)
        print("-------------------------")
        # print(loss)
        return loss

if __name__ == "__main__":
    # IMG_PATH = "grayscale.png"
    # IMG_PATH = "grayscale.png"
    IMG_PATH = "test_depth_imgs/mi_140.png"
    loss = Disparity_Smooth(IMG_PATH).calculate()
    print(loss)