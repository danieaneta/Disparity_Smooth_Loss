from disparity_smooth import Disparity_Smooth
import math 
import cv2 
import numpy 


print("----UNIT TEST START----")

IMG_PATH='grayscale.png'

#------------------------------
#read_image
#------------------------------

def test_read_image_run():
    Disparity_Smooth(IMG_PATH).read_image()
    assert True

def test_read_image():
    img, shape = Disparity_Smooth(IMG_PATH).read_image()
    assert type(img) == numpy.ndarray
    assert type(shape) == tuple

img, shape = Disparity_Smooth(IMG_PATH).read_image()
print("IMAGE: ", img)
print("SHAPE: ", shape)
    
#------------------------------
#get_parimeter
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
print("HEIGHT: ", height)
print("WIDTH: ", width)
print("HEIGHT_LIST: ", height_list)
print("WIDTH_LIST: ", width_list)

#------------------------------
#create_index_matrix
#------------------------------

def test_create_index_matrix_run():
    Disparity_Smooth(IMG_PATH).create_index_matrix(height_list, width_list)
    assert True

def test_create_index_matrix():
    matrix = Disparity_Smooth(IMG_PATH).create_index_matrix(height_list, width_list)
    assert type(matrix) == list

matrix = Disparity_Smooth(IMG_PATH).create_index_matrix(height_list, width_list)
print("MATRIX: ", matrix)

#------------------------------
#organize_matrix_lists
#------------------------------

def test_organize_matrix_lists_run():
    Disparity_Smooth(IMG_PATH).organize_matrix_lists(matrix, height_list, width)
    assert True

def test_organize_matrix_lists():
    matrix_master = Disparity_Smooth(IMG_PATH).organize_matrix_lists(matrix, height_list, width)
    assert type(matrix_master) == list


matrix_master = Disparity_Smooth(IMG_PATH).organize_matrix_lists(matrix, height_list, width)
print("MATRIX_MASTER: ", matrix_master)

# #------------------------------
# #h_end_pixel_list
# #------------------------------

# def test_h_end_pixel_list_run():
#     Disparity_Smooth(IMG_PATH).h_end_pixel_list(matrix_master, height_list)
#     assert True

# def test_h_end_pixel_list():
#     h_end_point_list = Disparity_Smooth(IMG_PATH).h_end_pixel_list(matrix_master, height_list)
#     assert type(h_end_point_list) == list
#     assert type(h_end_point_list[0]) == list
#     assert type(h_end_point_list[0][0]) == int

# h_end_point_list = Disparity_Smooth(IMG_PATH).h_end_pixel_list(matrix_master, height_list)

# print("H_END_POINT_LIST: ", h_end_point_list)

# #------------------------------
# #horizontal_diff
# #------------------------------

# def test_horizontal_diff_run():
#     Disparity_Smooth(IMG_PATH).horizontal_diff(img, matrix_master, height_list, width_list, h_end_point_list)
#     assert True

# def test_horizontal_diff():
#     h_diff = Disparity_Smooth(IMG_PATH).horizontal_diff(img, matrix_master, height_list, width_list, h_end_point_list)
#     assert type(h_diff)
#     assert type(h_diff[0]) == numpy.uint8

# h_diff = Disparity_Smooth(IMG_PATH).horizontal_diff(img, matrix_master, height_list, width_list, h_end_point_list)
# print("HORIZONTAL_DIFFERENCE: ", h_diff)

# #------------------------------
# #v_end_pixel_list
# #------------------------------

# def test_v_end_pixel_list_run():
#     Disparity_Smooth(IMG_PATH).v_end_pixel_list(matrix_master)
#     assert True

# def test_v_end_pixel_list():
#     v_end_point_list = Disparity_Smooth(IMG_PATH).v_end_pixel_list(matrix_master)
#     assert type(v_end_point_list) == list
#     assert type(v_end_point_list[0]) == list
#     assert type(v_end_point_list[0][0]) == int

# def v_end_pixel_list():
#     v_end_point_list = Disparity_Smooth(IMG_PATH).v_end_pixel_list(matrix_master)
#     return v_end_point_list


# v_end_point_list = Disparity_Smooth(IMG_PATH).v_end_pixel_list(matrix_master)
# print("V_END_PIXEL_LIST: ", v_end_pixel_list)

# #------------------------------
# #vertical_diff
# #------------------------------

# def test_vertical_diff_run():
#     Disparity_Smooth(IMG_PATH).vertical_diff(img, matrix_master, height_list, width_list, v_end_point_list)
#     assert True

# def test_vertical_diff():
#     v_diff = Disparity_Smooth(IMG_PATH).vertical_diff(img, matrix_master, height_list, width_list, v_end_point_list)
#     assert type(v_diff) == list
#     assert type(v_diff[0]) == numpy.uint8

# v_diff = Disparity_Smooth(IMG_PATH).vertical_diff(img, matrix_master, height_list, width_list, v_end_point_list)
# h_intensity_diff = h_diff
# v_intensity_diff = v_diff

# print("V_DIFF: ", v_diff)
# print("H_INTENSITY_DIFF: ", h_intensity_diff)
# print("V_INTENSITY_DIFF: ", v_intensity_diff)

# #------------------------------
# #horizontal_term_x
# #------------------------------

# def test_horizontal_term_x_run():
#     Disparity_Smooth(IMG_PATH).horizontal_term_x(h_diff, height_list, h_intensity_diff)
#     assert True

# def test_horizontal_term_x():
#     h_total = Disparity_Smooth(IMG_PATH).horizontal_term_x(h_diff, height_list, h_intensity_diff)
#     assert type(h_total) == numpy.float64

# h_total = Disparity_Smooth(IMG_PATH).horizontal_term_x(h_diff, height_list, h_intensity_diff)
# print("H_TOTAL: ", h_total)
# print("H_TOTAL_TYPE: ", type(h_total))

# #------------------------------
# #vertical_term_x
# #------------------------------

# def test_vertical_term_y_run():
#     Disparity_Smooth(IMG_PATH).vertical_term_y(v_diff, width_list, v_intensity_diff)
#     assert True

# def test_vertical_term_y():
#     v_total = Disparity_Smooth(IMG_PATH).vertical_term_y(v_diff,width_list, v_intensity_diff)
#     assert type(v_total) == numpy.float64

# v_total = Disparity_Smooth(IMG_PATH).vertical_term_y(v_diff, width_list, v_intensity_diff)
# print("V_TOTAL: ", v_total)
# print("V_TOTAL_TYPE: ", type(v_total))

# #------------------------------
# #loss_calc
# #------------------------------

# def test_loss_calc_run():
#     Disparity_Smooth(IMG_PATH).loss_calc(h_total, v_total)
#     assert True

# def test_loss_calc():
#     loss = Disparity_Smooth(IMG_PATH).loss_calc(h_total, v_total)
#     assert type(loss) == numpy.float64

# loss = Disparity_Smooth(IMG_PATH).loss_calc(h_total, v_total)
# print("LOSS: ", loss)

# #------------------------------
# #calculate
# #------------------------------
# def test_calculate_run():
#     Disparity_Smooth(IMG_PATH).calculate()
#     assert True

# def test_calcuate():
#     loss = Disparity_Smooth(IMG_PATH).calculate()
#     assert type(loss) == numpy.float64

# loss = Disparity_Smooth(IMG_PATH).calculate()
# print("LOSS: ", loss )
# print("----UNIT TEST END----")