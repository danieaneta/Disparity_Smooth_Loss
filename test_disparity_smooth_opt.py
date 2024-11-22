import math 
import numpy
import cv2 
from v02_Logic_Testing import Disparity_Smooth

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

#------------------------------
#v_end_pixel_array
#------------------------------

def test_v_end_pixel_array_run():
    Disparity_Smooth(IMG_PATH).v_end_pixel_array(width_list, height_list)
    assert True

def test_v_end_pixel_array():
    v_end_array = Disparity_Smooth(IMG_PATH).v_end_pixel_array(width_list, height_list)
    assert type(v_end_array) == numpy.ndarray

v_end_array = Disparity_Smooth(IMG_PATH).v_end_pixel_array(width_list, height_list)
print(v_end_array)

#------------------------------
#horizontal_diff
#------------------------------

def test_horizontal_diff_run():
    Disparity_Smooth(IMG_PATH).horizontal_diff(img, width_list, height_list, h_end_array)
    assert True

def test_horizontal_diff():
    h_diff_arr = Disparity_Smooth(IMG_PATH).horizontal_diff(img, width_list, height_list, h_end_array)
    assert type(h_diff_arr) == numpy.ndarray
h_diff_arr = Disparity_Smooth(IMG_PATH).horizontal_diff(img, width_list, height_list, h_end_array)
print(h_diff_arr)


#------------------------------
#vertical_diff
#------------------------------

def test_vertical_diff_run():
    Disparity_Smooth(IMG_PATH).vertical_diff(img, height_list, width_list, v_end_array)
    assert True

def test_vertical_diff():
    v_diff_arr = Disparity_Smooth(IMG_PATH).vertical_diff(img, height_list, width_list, v_end_array)
    assert type(v_diff_arr) == numpy.ndarray

v_diff_arr = Disparity_Smooth(IMG_PATH).vertical_diff(img, height_list, width_list, v_end_array)
print(v_diff_arr)

#------------------------------
#horizontal_term_x
#------------------------------

h_intensity_diff = h_diff_arr

def test_horiztonal_term_x_run():
    Disparity_Smooth(IMG_PATH).horizontal_term_x(h_diff_arr, height_list, h_intensity_diff)
    assert True

def test_horizontal_term_x():
    h_total = Disparity_Smooth(IMG_PATH).horizontal_term_x(h_diff_arr, height_list, h_intensity_diff)
    assert type(h_total) == numpy.float64

h_total = Disparity_Smooth(IMG_PATH).horizontal_term_x(h_diff_arr, height_list, h_intensity_diff)
print(h_total)


#------------------------------
#vertical_term_x
#------------------------------

v_intensity_diff = v_diff_arr

def test_vertical_term_x_run():
    Disparity_Smooth(IMG_PATH).vertical_term_y(v_diff_arr, width_list, v_intensity_diff)
    assert True

def test_vertical_term_x():
    v_total = Disparity_Smooth(IMG_PATH).vertical_term_y(v_diff_arr, width_list, v_intensity_diff)
    assert type(v_total) == numpy.float64

v_total = Disparity_Smooth(IMG_PATH).vertical_term_y(v_diff_arr, width_list, v_intensity_diff)
print(v_total)

#------------------------------
#loss_calc
#------------------------------

def test_loss_calc_run():
    Disparity_Smooth(IMG_PATH).loss_calc(h_total, v_total)
    assert True

def test_loss_calc():
    loss = Disparity_Smooth(IMG_PATH).loss_calc(h_total, v_total)
    assert type(loss) == numpy.float64

loss = Disparity_Smooth(IMG_PATH).loss_calc(h_total, v_total)
print(loss)

#------------------------------
#calculate
#------------------------------

def test_calculate_run():
    Disparity_Smooth(IMG_PATH).calculate()
    assert True

def test_calculate():
    loss = Disparity_Smooth(IMG_PATH).calculate()
    assert type(loss) == numpy.float64

print(loss)