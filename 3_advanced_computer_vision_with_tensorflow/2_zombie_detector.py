'''
Installs

tf-keras
tensorflow
keras
'''

# imports 
import os,re,time,json,zipfile
import PIL.Image,PIL.ImageFont,PIL.ImageDraw
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_datasets tfds
import cv2

def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax,color=(250,0,0),thickness=5):

def draw_bounding_boxes_on_image():

def draw_bounding_boxes_on_image_array():

def display_digits_with_boxes():

def read_image_tfds():

def read_image_with_shape():

def read_image_tfds_with_original_bbox(data):

def dataset_to_numpy_util():

def dataset_to_numpy_with_original_bboxes_util():

def get_visualization_training_dataset():

def get_visualization_validation_dataset():

def get_training_dataset():

def get_validation_dataset():

def feature_extractor():

def dense_layers():

def bounding_box_regression():

def final_model():

def define_and_compile_model():

def intersection_over_union(pred_box, true_box):

    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box,4,axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box,4,axis)

    xmin_overlap = np.maximum()
    xmax_overlap = np.minimum()
    ymin_overlap = np.maximum()
    ymax_overlap = np.minimum()

    pred_box_area = (xmax_pred - xmin_pred) * 



def intersection_over_union():
