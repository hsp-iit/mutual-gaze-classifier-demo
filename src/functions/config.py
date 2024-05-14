#!/usr/bin/python3

# image
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# open pose
OPENPOSE_JOINTS_POSE = [0, 15, 16, 17, 18]
MMPOSE_JOINTS_POSE = [2, 3, 4, 5, 6]
OPENPOSE_JOINTS_FACE = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 68, 69] # joint indices for openpose
MMPOSE_JOINTS_FACE = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] # coco wholebody no nose
# MMPOSE_JOINTS_FACE = [27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] # coco wholebody (nose and eyes) 

def get_keypoints_inds(keypoint_detector):
    JOINTS_POSE = []
    JOINTS_FACE = []
    if keypoint_detector == "openpose":
        JOINTS_POSE = OPENPOSE_JOINTS_POSE
        JOINTS_FACE = OPENPOSE_JOINTS_FACE
    elif keypoint_detector == "mmpose":
        JOINTS_POSE = MMPOSE_JOINTS_POSE
        JOINTS_FACE = MMPOSE_JOINTS_FACE
    else:
        raise Exception("Unrecognized keypoint detector")
    
    return JOINTS_POSE, JOINTS_FACE
