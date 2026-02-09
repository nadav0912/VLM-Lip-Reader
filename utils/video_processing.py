import math
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import FACE_ANCHORS, LIPS_INDICES



def extract_face_data(landmarks, frame_w, frame_h):
    """
    Extracts the full face data for future analysis.
    Converts normalized coordinates (0-1) to pixels (Absolute).
    """
    
    # 1. Get the 6 anchors and convert to pixels
    anchors_data = {}
    for name, idx in FACE_ANCHORS.items():
        lm = landmarks[idx]
        
        # lm.x is between 0 and 1, frame_w is like 1920
        pixel_x = int(lm.x * frame_w)
        pixel_y = int(lm.y * frame_h)
        
        anchors_data[name] = [pixel_x, pixel_y]

    return anchors_data


def get_mouth_score(landmarks):
    # Calculate the average confidence of the lips
    scores = []
    for idx in LIPS_INDICES:
        lm = landmarks[int(idx)]
        # Check confidence (visibility or presence)
        s = getattr(lm, 'visibility', getattr(lm, 'presence', 1.0))
        scores.append(s)
    
    avg_score = sum(scores) / len(scores)

    return avg_score


def calc_face_size_ratio(landmarks):
    """
    # Calculate the face size ratio
    # note: landmarks point are already normalized to the image size
    """
    # We use the distance between the nose and the ears
    x_coords = [lm.x for lm in landmarks]
    min_x, max_x = min(x_coords), max(x_coords)
    face_width = max_x - min_x
    return face_width


def calc_yaw_angle(landmarks):
    """
    # Calculate the yaw angle
    # Calculates the horizontal symmetry ratio between the nose and ears to estimate head yaw (left/right rotation).
    """
    # We use the distance between the nose and the ears
    nose_x = landmarks[1].x
    left_ear_x = landmarks[234].x
    right_ear_x = landmarks[454].x
    return (nose_x - left_ear_x) / (right_ear_x - nose_x) + 1e-6