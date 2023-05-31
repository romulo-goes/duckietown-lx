from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_left = np.random.rand(*shape)
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    # res[:, int(shape[1]/2):shape[1]] = -0.1
    res[240:480, 250:320] = -1
    # res[shape[0]-200:shape[0], 80:int(shape[1]/2)+5] = 1
    steer_matrix_left = res
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_right = np.random.rand(*shape)
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    # res[:, int(shape[1]/2):shape[1]] = -0.1
    res[240:480, 320:370] = 1
    steer_matrix_right = res
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
        right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    # TODO: implement your own solution here
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = 5

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    top = np.zeros([int(np.floor(h/2)),w])
    bottom = np.ones([int(np.floor(h/2)),w])
    mask_ground = np.concatenate((top,bottom))

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    width = image.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0
# Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

# Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    threshold = 25 # CHANGE ME

    mask_mag = (Gmag > threshold)
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    white_lower_hsv = np.array([40, 0, 100])         # CHANGE ME
    white_upper_hsv = np.array([150, 200, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([20, 0, 100])        # CHANGE ME
    yellow_upper_hsv = np.array([40, 255, 255])  # CHANGE ME

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_left_edge = mask_ground * mask_left * mask_mag * mask_yellow * mask_sobelx_neg * mask_sobely_neg
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_white * mask_sobelx_pos * mask_sobely_neg
    # mask_left_edge = np.random.rand(h, w)
    # mask_right_edge = np.random.rand(h, w)

    return mask_left_edge, mask_right_edge
