# Face anchors (mediapipe indices)
FACE_ANCHORS = {
    "eye_l": 33,    # Left eye (to align angle)
    "eye_r": 263,   # Right eye (to align angle)
    "mouth_l": 61,  # Left corner (to calculate width)
    "mouth_r": 291, # Right corner (to calculate width)
    "mouth_t": 0,   # Upper lip (to calculate height/center)
    "mouth_b": 17   # Lower lip (to calculate height/center)
}

# All lips indices(mediapipe indices) to keep
LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 
    14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415
]

# Frame statuses
FRAME_STATUS = {
    2: "silence",
    1: "speaking",
    0: "other speaker",
    -1: "movement" 
}

# Analysis video shortcuts meanings
ANALYSIS_VIDEO_SHORTCUTS = {
    'i': 'index',
    't': 'time',
    's': 'status',
    'a': 'anchors',
    'r': 'reject reason'
}