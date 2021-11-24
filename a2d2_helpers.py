# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import cv2
import numpy as np
import numpy.linalg as la
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
import matplotlib.pyplot as plt


EPSILON = 1.0e-10

def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']
     
    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)
    
    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")
        
    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm
    
    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)
 
    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)
    
    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)
    
    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")
        
    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm
    
    return x_axis, y_axis, z_axis


def get_origin_of_a_view(view):
    return view['origin']


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)
    
    # get origin 
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)
    
    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis
    
    # origin
    transform_to_global[0:3, 3] = origin
    
    return transform_to_global

def undistort_image(image, cam_name, config):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
                  np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        
        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                      D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                      distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image
    

def a2d2_box_to_transform_matrix_and_size(label: dict):
    """Converts from an A2D2 cuboid label to a transform matrix"""
    x, y, z = label["center"]
    l, w, h = label["size"]
    rot_ang = label["rot_angle"]
    axis = label["axis"]
    # A2D2 uses an axis angle convention, see https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    rot = pr.matrix_from_axis_angle([axis[0], axis[1], axis[2], rot_ang])
    qw, qx, qy, qz = pr.quaternion_from_matrix(rot)
    return pt.transform_from_pq([x, y, z, qw, qx, qy, qz]), [l, w, h]

def draw_2d_bounding_box(image, bbox, label, color):
    """Draws a 2D bounding box using OpenCV"""
    x0, y0, x1, y1 = np.rint(bbox).astype(int)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, label, (x0,y0-10), 0, 0.5, color, 2)
    
def generate_color_map(labels_3d):
    """Generates colors to disambiguate classes"""
    label_class_names = sorted(list(set([label["class"] for label in labels_3d.values()])))
    cmap = plt.get_cmap("Set3")
    colors = cmap(np.linspace(0, 1, len(label_class_names)))
    return {label_name: 255*colors[i] for i, label_name in enumerate(label_class_names)}


def x_forward_to_z_foward(transform):
    pq = pt.pq_from_transform(transform)
    rot = pr.matrix_from_quaternion(pq[3:])

    new_x = rot @ np.array([0, -1, 0])
    new_y = rot @ np.array([0, 0, -1])
    rot = pr.matrix_from_two_vectors(new_x, new_y)
    pq[3:] = pr.quaternion_from_matrix(rot)
    return pt.transform_from_pq(pq)
    