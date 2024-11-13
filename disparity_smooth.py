import math 
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from PIL import Image

grayscale_matrix = np.array([
    [50, 100, 150],
    [200, 250, 25],
    [75, 125, 175]
], dtype=np.uint8)


cv2.imwrite('grayscale.png', grayscale_matrix)
image = cv2.imread('grayscale.png', cv2.IMREAD_GRAYSCALE)

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
    
    def create_index_matrix(self, height_list, width_list):
        matrix = []
        for c in height_list:
            for i in width_list:
                matrix_key = [c, i]
                matrix.append(matrix_key)
        return matrix
    
    def organize_matrix_lists(self, matrix, height_list, width): 
        matrix_master = []
        for i in height_list:
            d_list = matrix[0:width]
            matrix_master.append(d_list)
            for i in d_list:
                matrix.remove(i)
        return matrix_master
    
    def h_end_pixel_list(self, matrix_master, height_list):
        h_end_list = []
        for i in height_list:
            end_point = matrix_master[i][-1]
            h_end_list.append(end_point)
        return h_end_list

    def horizontal_diff(self, img, matrix_master, height_list, width_list, h_end_point_list):
        h_diff = []
        for c in height_list:
            for r in width_list:
                if matrix_master[c][r] in h_end_point_list:
                    secondary_index_h, secondary_index_v = matrix_master[c][r-1][0], matrix_master[c][r-1][1]
                    secondary_pixel = img[secondary_index_h, secondary_index_v]
                else:
                    secondary_index_h, secondary_index_v = matrix_master[c][r+1][0], matrix_master[c][r+1][1]
                    secondary_pixel = img[secondary_index_h, secondary_index_v]
                primary_pixel_h, primary_pixel_v = matrix_master[c][r][0], matrix_master[c][r][1]
                main_pixel = img[primary_pixel_h, primary_pixel_v]
                difference = secondary_pixel - main_pixel
                h_diff.append(difference)
        return h_diff

    def v_end_pixel_list(self, matrix_master):
        v_end_list = matrix_master[-1]
        return v_end_list

    def vertical_diff(self, img, matrix_master, height_list, width_list, v_end_point_list):
        v_diff = []
        for r in width_list:
            for c in height_list:
                if matrix_master[c][r] in v_end_point_list:
                    secondary_index_h, secondary_index_v = matrix_master[c-1][r][0], matrix_master[c-1][r][1]
                    secondary_pixel = img[secondary_index_h, secondary_index_v]
                else:
                    secondary_index_h, secondary_index_v = matrix_master[c+1][r][0], matrix_master[c+1][r][1]
                    secondary_pixel = img[secondary_index_h, secondary_index_v]
                primary_pixel_h, primary_pixel_v = matrix_master[c][r][0], matrix_master[c][r][1]
                main_pixel = img[primary_pixel_h, primary_pixel_v]
                difference = secondary_pixel - main_pixel
                v_diff.append(difference) 
        return v_diff

    def horizontal_term_x(self, h_diff, height_list, h_intensity_diff):
        #take h_diff(n,n) * e^-I(n,n)

        h_total = []
        for c in height_list:
            pixel_total = h_diff[c] * math.e ** h_intensity_diff[c]
            h_total.append(pixel_total)

        return h_total

    def vertical_term_y(self, v_diff, width_list, v_intensity_diff):
        v_total = []

        for r in width_list:
            pixel_total = v_diff[r] * math.e ** v_intensity_diff[r]
            v_total.append(pixel_total)
        return v_total
    
    def horizontal_total_sum(h_total):
        # h_total = float
        # for c in height_list:
        #     for r in width_list:
        #         h_total = h_total + h_total[c][r]
        h_total_sum  = h_total
        return h_total_sum

    def vertical_total_sum(v_total, height_list, width_list):
        v_total = float
        for r in width_list:
            for c in height_list:
                v_total = v_total + v_total[c][r]
        return v_total

    def loss_calc(self, h_total, v_total):
        loss = sum(h_total + v_total)
        return loss

    def calculate(self):
        img, shape = self.read_image()
        height, width, height_list, width_list = self.get_perimeter(shape)
        matrix = self.create_index_matrix(height_list, width_list)
        matrix_master = self.organize_matrix_lists(matrix, height_list, width)
        h_end_point_list = self.h_end_pixel_list(matrix_master, height_list)
        h_diff = self.horizontal_diff(img, matrix_master, height_list, width_list, h_end_point_list)
        v_end_point_list = self.v_end_pixel_list(matrix_master)
        v_diff = self.vertical_diff(img, matrix_master, height_list, width_list, v_end_point_list)
        h_intensity_diff = h_diff
        v_intensity_diff = v_diff
        h_total = self.horizontal_term_x(h_diff, height_list, h_intensity_diff)
        v_total = self.vertical_term_y(v_diff, width_list, v_intensity_diff)
        # horizontal_total_sum = self.horizontal_total_sum(h_total, height_list, width_list)
        # vertical_total_sum = self.vertical_total_sum(v_total, height_list, width_list)
        # loss = self.loss_calc(self, horizontal_total_sum, vertical_total_sum)

        # print(loss)

if __name__ == "__main__":
    IMG_PATH='grayscale.png'
    Disparity_Smooth(IMG_PATH).horizontal_total_sum(5, 5, 5)
