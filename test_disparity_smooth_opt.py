import math 
import numpy
import cv2 
from disparity_smooth_optimization import Disparity_Smooth

IMG_PATH = 'grayscale.png'
 
#------------------------------
#read_image
#------------------------------

def test_read_image_run():
    Disparity_Smooth(IMG_PATH).read_image()
    assert True


def test_read_image():
    img, shape = Disparity_Smooth(IMG_PATH).read_image()
    assert type(img) == numpy.ndarray

img, shape = Disparity_Smooth(IMG_PATH).read_image()


#------------------------------
#get_perimeter
#------------------------------

def test_get_perimeter_run():
    Disparity_Smooth(IMG_PATH).get_perimeter(shape)
    assert True

def test_get_perimeter():
    height, width, height_list, width_list = Disparity_Smooth(IMG_PATH).get_perimeter(shape)
    assert type(height) == int
    assert type(width) == int
    assert type(height_list) == range
    assert type(width_list) == range

height, width, height_list, width_list = Disparity_Smooth(IMG_PATH).get_perimeter(shape)

#------------------------------
#h_end_pixel_array
#------------------------------

def test_h_end_pixel_array_run():
    Disparity_Smooth(IMG_PATH).h_end_pixel_array(width_list, height_list)
    assert True

def test_h_end_pixel_array_():
    h_end_array = Disparity_Smooth(IMG_PATH).h_end_pixel_array(width_list, height_list)
    assert type(h_end_array) == numpy.ndarray

h_end_array = Disparity_Smooth(IMG_PATH).h_end_pixel_array(width_list, height_list)
print(h_end_array)



# test = (2,0)
# if any(numpy.array_equal(test, arr) for arr in h_end_array):
#     print("Yes")


#------------------------------
#horizontal_diff
#------------------------------

def test_horizontal_diff_run():
    Disparity_Smooth(IMG_PATH).horizontal_diff(img, width_list, height_list, h_end_array)
    assert True

def test_horizontal_diff():
    h_diff_arr = Disparity_Smooth(IMG_PATH).horizontal_diff(img, width_list, height_list, h_end_array)
    assert type(h_diff_arr) == str

h_diff_arr = Disparity_Smooth(IMG_PATH).horizontal_diff(img, width_list, height_list, h_end_array)
print(h_diff_arr)