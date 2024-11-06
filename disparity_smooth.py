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

line_image = np.ones((3, 3))  # Start with a white image
line_image[:, 1] = 0  # Set the middle column to black to create the dark line

# cv2.imwrite('test.jpg', line_image)

image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
# image = Image.open('test.jpg')

# print(image[0, 0, 0])
# print(image[1])
# print(image[2])
print(image.shape)
#rows, columns #channels
#color 3 [0, 0, 0]
#grayscale 3 [0, 0, 0] 

