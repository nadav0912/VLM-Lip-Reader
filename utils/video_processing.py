import math
import numpy as np
import os
import cv2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import FACE_ANCHORS, LIPS_INDICES

def calc_movement_metrics(curr_lm, prev_lm, curr_yaw, prev_yaw, curr_ratio, prev_ratio):
    """
    Calculates the change between the current and previous landmarks.
    Returns: (movement distance, yaw angle change, face size change)
    """
    if prev_lm is None:
        return 0.0, 0.0, 0.0

    # 1. Calculate the movement distance (by the nose - point 1 in MediaPipe)
    # Since the points are normalized (0-1), the distance is in percentage of the screen
    c_nose = curr_lm[1]
    p_nose = prev_lm[1]
    dist = math.sqrt((c_nose.x - p_nose.x)**2 + (c_nose.y - p_nose.y)**2)

    # 2. Calculate the yaw angle change (in absolute value)
    yaw_diff = abs(curr_yaw - prev_yaw)

    # 3. Calculate the face size change (in absolute value)
    size_diff = abs(curr_ratio - prev_ratio)

    return dist, yaw_diff, size_diff


def interpolate_point(p1, p2, alpha):
    """
    Calculates a new point that is located at a relative distance (alpha) between p1 and p2.
    alpha = 0.5 -> exactly in the middle.
    alpha = 0.33 -> one third of the way from p1.
    """
    return [
        p1[0] * (1 - alpha) + p2[0] * alpha,  # X
        p1[1] * (1 - alpha) + p2[1] * alpha   # Y
    ]

def fill_missing_landmarks(landmarks_map):
    """
    Receives a dictionary of {index: landmarks}.
    Returns a new dictionary in which the holes (up to 2 consecutive) are filled.
    Frames missing at the beginning or end are not filled (muted).
    """
    if not landmarks_map:
        return {}

    # 1. Create a copy and find the boundaries
    filled_map = landmarks_map.copy()
    existing_indices = sorted(landmarks_map.keys())
    
    start_frame = existing_indices[0]
    end_frame = existing_indices[-1]

    # 2. Loop through all the frames in the range (from the first one we have to the last one we have)
    for i in range(start_frame, end_frame + 1):
        
        # If the frame exists - skip
        if i in filled_map:
            continue
            
        # --- We got a hole! ---
        
        # Search for the previous neighbor (we go back until we find)
        prev_idx = i - 1
        while prev_idx not in filled_map and prev_idx >= start_frame:
            prev_idx -= 1
            
        # Search for the next neighbor (we go forward until we find)
        next_idx = i + 1
        while next_idx not in filled_map and next_idx <= end_frame:
            next_idx += 1
            
        # If we found both a parent and a child (neighbors on both sides)
        if prev_idx in filled_map and next_idx in filled_map:
            
            # Calculate the distance and relative position
            total_gap = next_idx - prev_idx
            current_step = i - prev_idx
            
            # Alpha: How close are we to the end? (between 0 and 1)
            # Example for a single hole: gap=2, step=1 -> alpha=0.5
            # Example for 2 holes: 
            #   First frame: gap=3, step=1 -> alpha=0.33
            #   Second frame:   gap=3, step=2 -> alpha=0.66
            alpha = current_step / float(total_gap)
            
            prev_lms = filled_map[prev_idx]
            next_lms = filled_map[next_idx]
            new_lms = {}
            
            # Interpolation for each body part (eyes, mouth, etc.)
            for key in prev_lms:
                # The point itself is a list [x, y]
                new_lms[key] = interpolate_point(prev_lms[key], next_lms[key], alpha)
            
            filled_map[i] = new_lms
            # print(f"🔧 Repaired frame {i} using {prev_idx} and {next_idx} (Alpha: {alpha:.2f})")

    return filled_map


def to_px(pt, img_width, img_height):
    # Convert normalized coordinates to pixels
    x, y = pt
    if x <= 1.0 and y <= 1.0:
        return float(x * img_width), float(y * img_height)
    return float(x), float(y)


def get_mouth_roi_params(landmarks, img_width, img_height, is_normalized=True, padding_scale=1.25, engale_by_eye=False):
    # Calculate the mouth ROI parameters
    
    ml = landmarks["mouth_l"]
    mr = landmarks["mouth_r"]             
    mt = landmarks["mouth_t"]
    mb = landmarks["mouth_b"]
    er = landmarks["eye_r"]
    el = landmarks["eye_l"]

    if is_normalized:
        ml = to_px(ml, img_width, img_height)
        mr = to_px(mr, img_width, img_height)
        mt = to_px(mt, img_width, img_height)
        mb = to_px(mb, img_width, img_height)
        er = to_px(er, img_width, img_height)
        el = to_px(el, img_width, img_height)
    
    # 2. Calculate the center
    center_x = (ml[0] + mr[0] + mt[0] + mb[0]) / 4.0
    center_y = (ml[1] + mr[1] + mt[1] + mb[1]) / 4.0
    
    # 3. Calculate the size of the square
    mouth_width = math.hypot(mr[0] - ml[0], mr[1] - ml[1])
    
    # The mouth is usually wider than higher, so the width is the determining factor
    roi_size = int(mouth_width * padding_scale)
    
    # Half size (to go left/right from the center)
    half_size = roi_size // 2
    
    # 4. Calculate the angle (Angle) - to align the head
    if engale_by_eye:
        dy = er[1] - el[1]
        dx = er[0] - el[0]
    else:
        # Calculate the angle by moth sides
        dy = mr[1] - ml[1]
        dx = mr[0] - ml[0]
    
    angle = math.degrees(math.atan2(dy, dx))

    # ((Center X, Center Y), (Width, Height), Angle)
    return ((center_x, center_y), (roi_size, roi_size), angle)


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
    # Calculate the yaw angle (head rotation 0-1)
    # Calculates the horizontal symmetry ratio between the nose and ears to estimate head yaw (left/right rotation).
    """
    # We use the distance between the nose and the ears
    nose_x = landmarks[1].x
    left_ear_x = landmarks[234].x
    right_ear_x = landmarks[454].x
    
    # Use abs to ensure positive distances always
    dist_to_left = abs(nose_x - left_ear_x)
    dist_to_right = abs(nose_x - right_ear_x)

    # Protect against division by zero (add epsilon to the denominator itself)
    if dist_to_right < 1e-6:
        return 100.0 # Extreme angle (profile)

    return dist_to_left / dist_to_right