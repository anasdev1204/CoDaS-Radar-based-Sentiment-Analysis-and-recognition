import os
from glob import glob


MOTION_MAP = {
    "M01": "lateral-raise",
    "M02": "push-down",
    "M03": "lift",
    "M04": "pull",
    "M05": "push",
    "M06": "lateral-to-front",
    "M07": "swipe-right",
    "M08": "swipe-left",
    "M09": "throw",
    "M10": "arms-swing",
    "M11": "two-hand-throw",
    "M12": "two-hand-push",
    "M13": "two-hand-pull",
    "M14": "two-hand-lateral-raise",
    "M15": "left-arm-circle",
    "M16": "right-arm-circle",
    "M17": "two-hand-outward-circles",
    "M18": "two-hand-inward-circles",
    "M19": "two-hand-lateral-to-front",
    "M20": "circle-clockwise",
    "M21": "circle-counter-clockwise",
}
ACTIVITY_MAP = {
    "A01": "walking",
    "A02": "running",
    "A03": "lying",
    "A04": "sitting",
    "A05": "ascending-stairs",
    "A06": "descending-stairs",
    "A07": "passing-basketball",
    "A08": "playing-badminton",
    "A09": "kick-floor-ball",
    "A10": "kick-football",
}
EMOTION_MAP = {
    "E01": "focus",
    "E02": "distraction",
    "E03": "stress",
    "E04": "relaxation",
    "E05": "depression",
    "E06": "excitement",
}

def map_to_name(behavior_code):
    return MOTION_MAP.get(behavior_code, ACTIVITY_MAP.get(behavior_code, EMOTION_MAP.get(behavior_code, "UNKNOWN")))

def behavior_name_from_path(path):
    return map_to_name([os.path.normpath(path).split(os.sep)[-2]])
