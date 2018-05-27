import random


def jitter_position(bbox, size, step=(0, 0)):

    while True:
        x_min_step = random.randint(-step[0], step[0])
        x_max_step = random.randint(-step[0], step[0])
        new_x_min = bbox[0] - x_min_step
        new_x_max = bbox[2] - x_max_step
        new_bbox = (new_x_min, bbox[1], new_x_max, bbox[3])
        if (_check_position(new_bbox, size)):
            break

    while True:
        y_min_step = random.randint(-step[1], step[1])
        y_max_step = random.randint(-step[1], step[1])
        new_y_min = bbox[1] - y_min_step
        new_y_max = bbox[3] - y_max_step
        new_bbox = (new_bbox[0], new_y_min, new_bbox[2], new_y_max)
        if (_check_position(new_bbox, size)):
            break



    return new_bbox

def _check_position(bbox, size):
    if ((bbox[2] - bbox[0]) <= 4) or ((bbox[3] - bbox[1]) <= 4):
        return False
    elif (bbox[0] < 0) or (bbox[1] < 0):
        return False
    elif (bbox[2] > size[1]) or (bbox[3] > size[0]):
        return False
    else:
        return True
