import os
import numpy as np
from detectron2.structures import BoxMode

def get_robotcar_dicts(data_root, split_path):
    with open(split_path, 'r') as f:
        image_list = [image_info.strip().split(' ') for image_info in f.readlines()]
    label_path = os.path.join(data_root, 'label_2d')

    dataset_dicts = []
    object_index = 1
    for image_info in image_list:
        image_index = int(image_info[0])
        label_name = os.path.join(label_path, image_info[1]+'.txt')
        with open(label_name, 'r') as f:
            label_list = [label_info.strip().split(' ') for label_info in f.readlines()]

        record = {}
        record['data_root'] = data_root
        record['timestamp'] = image_info[1]
        record['image_id'] = image_index

        cars = []
        for label_info in label_list:
            car_index = float(label_info[1])
            xc = float(label_info[2])
            yc = float(label_info[3])
            width = float(label_info[4])
            height = float(label_info[5])
            yaw = float(label_info[6])
            assert yaw >= -180 and yaw < 180, 'rotation angles of labels should be within [-180,180).'
            car_instance = {
                'bbox': [xc, yc, width, height, yaw],
                'bbox_mode': BoxMode.XYWHA_ABS,
                'category_id': 0,
                'iscrowd': 0,
                'id': object_index,
                'car_id': car_index
            }
            cars.append(car_instance)
            object_index += 1
        record['annotations'] = cars
        
        dataset_dicts.append(record)

    return dataset_dicts

