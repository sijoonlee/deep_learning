import cv2
import numpy as np

def preprocessing(img, height, width):
    preprocessed_image = cv2.resize(img, (width, height))
    preprocessed_image = preprocessed_image.transpose((2,0,1)) #bgr
    preprocessed_image = preprocessed_image.reshape(1,3,height, width)
    return preprocessed_image

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the pose estimation model
    return preprocessing(preprocessed_image, 256, 456)


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the text detection model
    return preprocessing(preprocessed_image, 768, 1280)


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model

    return preprocessing(preprocessed_image, 72, 72)
