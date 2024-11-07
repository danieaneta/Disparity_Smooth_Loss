import math 
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from PIL import Image


'''
Take a matrix, and select a pixel in the matrix > 1st pixel selection will be (1,1)
take in surroundings/perimeter >
    store "perimeter" = [] 
    left, right, top, bottom = [-1, 0], [1, 0], [0, 1], [0, -1]                
for pixel in matrix: 
    if left (-1)is None:
        append 0 (EMPTY)
    else:
        append 1 (NOT-EMPTY)
        
    if right (+1) is None:
        append 0 (EMPTY)
    else:
        append 1 (NOT-EMPTY)
        
    if top (+1) is None:
        append 0 (EMPTY)
    else: 
        append 1 (NOT-EMPTY)
        
    if bottom (-1) is None:
        append 0 (EMPTY)
    else:
        append 1 (NOT-EMPTY)
            
ERROR CHECK: check if perimeter list len = 4
    if not 4: 
        perimeter = []
        go back and redo checks 
    else:
        continue
        
if list has one 0:
    figure out where zero is
    process
if list has 2 zero: 
    figure out where zeros are
    process
    
    
    
    
create a matrix equal to dimensions of image
look through perimeter list
    if list has one 0:
        figure out where zero is
        process
    if list has 2 zero: 
        figure out where zeros are
        process

processes 

'''

line_image = np.ones((10, 20, 3), dtype=np.uint8) * 255  # Start with a white image
cv2.imwrite('test_white.png', line_image)

image = cv2.imread('test_white.png', cv2.IMREAD_GRAYSCALE)
# image = Image.open('test.jpg')

# print(image[0, 0, 0])
# print(image[1])
# print(image[2])
# print(image)

height, width = image.shape[0], image.shape[1]
total_pixels = height * width
height_list, width_list = range(height), range(width)
#need rows

matrix = []

for c in range(height):
    for i in range(width):
        matrix_key = [c, i]
        matrix.append(matrix_key)

# print(matrix)

matrix_lists = []
for i in range(height):
    #divide width
    d_list = matrix[0:width] 
    matrix_lists.append(d_list)
    for i in d_list:
        matrix.remove(i)

# print(matrix_lists[1][0])

print(matrix_lists[0][-1])

for c in range(height):
    end_pixel = matrix_lists[c][-1]



# for c in range(height):
#     for i in range(width):
#         print(matrix_lists[c][i])

 


# for i in height_list:
#     print(i)



# print(height, width)
#rows, columns #channels
#color 3 [0, 0, 0]
#grayscale 3 [0, 0, 0] 

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
    
    def organize_matrix_lists(self, matrix, height_list): 
        matrix_master = []
        for i in height_list:
            d_list = matrix[0:width]
            matrix_master.append(d_list)
            for i in d_list:
                matrix.remove(i)
        return matrix_master
    
    def end_pixel_list(self, matrix_master):
        pass

    def horizontal_process(self, img, matrix_master, height_list, width_list):
        h_diff = []
        for c in height_list:
            for r in width_list:
                if matrix_master[c][r] == 3:
                    secondary_pixel = img[matrix_master[c][r-1]]
                else:
                    secondary_pixel = img[matrix_master[c][r+1]]
                main_pixel = img[matrix_master[c][r]]
                difference = secondary_pixel - main_pixel
                h_diff.append(difference)
        return h_diff

    def vertical_proccess(self):
        pass


    def calculate(self):
        img, shape = self.read_image()
        height, width, height_list, width_list = self.get_perimeter(shape)
        matrix = self.create_index_matrix(height_list, width_list)
        matrix_master = self.organize_matrix_lists(matrix, height_list)
        h_diff = self.horizontal_process(img, matrix_master, height_list, width_list)
