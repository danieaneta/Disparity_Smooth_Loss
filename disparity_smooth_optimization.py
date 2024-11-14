import math 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

"""
Ideas: 
Do not make matrix, just use h_end and v_end list
Store matrix in numpy array
"""


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
        return height, width, height_list, width_list
    
    
    def h_end_pixel_array(self, width_list, height_list):
        h_end_ar = []
        for c in height_list:
            end_point = np.array((c, width_list[-1]))
            h_end_ar.append(end_point)
        h_end_ar_arr = np.empty(shape=len(h_end_ar), dtype=object)
        h_end_ar_arr[:] = h_end_ar
        return h_end_ar_arr

    def v_end_pixel_array(self, width_list, height_list):
        v_end_ar = []
        for r in width_list:
            end_point = np.array((height_list[-1], r))
            v_end_ar.append(end_point)
        v_end_ar_arr = np.empty(shape=len(v_end_ar), dtype=object)
        v_end_ar_arr[:] = v_end_ar
        return v_end_ar_arr
    

    def horizontal_diff(self, img, height_list, width_list, h_end_points):
        h_diff = []
        for c in height_list:
            for r in width_list:
                main_pixel_index = (c, r)
                if any(np.array_equal(main_pixel_index, arr) for arr in h_end_points):
                    print("Hit End Point....")
                    secondary_pixel_index = (c, (r-1))
                    print(secondary_pixel_index)
                else: 
                    print("Start or Middle Pixel Point...")
                    secondary_pixel_index = (c, (r+1))
                    print(secondary_pixel_index)

                # print(img[secondary_pixel_index])
                # print(img[c, r])

                second_pixel, main_pixel = float(img[secondary_pixel_index]), float(img[c, r])
                difference = float(second_pixel - main_pixel)
                diff_np = np.array(difference)
                print(diff_np)
                h_diff.append(diff_np)

        h_diff_arr = np.empty(shape=len(h_diff), dtype=object)
        h_diff_arr[:] = h_diff
        return h_diff_arr

    def vertical_diff(self, img, height_list, width_list, v_end_points):
        v_diff = []
        for r in width_list:
            for c in height_list:
                if img[c][r] in v_end_points:
                    secondary_pixel = img[c-1][r]
                else:
                    secondary_pixel = img[c+1][r]
                difference = secondary_pixel - img[c][r]
                diff_np = np.array(difference)
                v_diff.append(diff_np)
        
        v_diff_arr = np.empty(shape=len(v_diff), dtype=object)
        return v_diff_arr


    def calculate(self):
        img, shape = self.read_image()
        height, width, height_list, width_list = self.get_perimeter(shape)
        h_end_points = self.h_end_pixel_array(width_list, height_list)
        v_end_points = self.v_end_pixel_array(width_list, height_list)
        h_diff_arr = self.horizontal_diff(img, height_list, width_list, h_end_points)
        v_diff_arr = self.vertical_diff(img, height_list, width_list, v_end_points)



