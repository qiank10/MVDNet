import copy
import logging
import numpy as np
import torch
import os
import random

from detectron2.data import detection_utils as utils
from detectron2.structures import Boxes, BoxMode, Instances
from .robotcar_utils import lidar_foggify, lidar_pc2pixor, annotations_to_instances_directed


class RobotCarMapper:
    
    def __init__(self, cfg, mode='train'):
        self.eval_beta = cfg.INPUT.EVAL_BETA
        self.history_on = cfg.INPUT.HISTORY_ON
        self.num_history = cfg.INPUT.NUM_HISTORY

        self.n = cfg.INPUT.LIDAR.FOG.N
        self.g = cfg.INPUT.LIDAR.FOG.G
        self.fog_ratio = cfg.INPUT.LIDAR.FOG.FOG_RATIO
        self.beta_range = cfg.INPUT.LIDAR.FOG.BETA_RANGE
        self.dmin = cfg.INPUT.LIDAR.FOG.D_MIN
        self.fraction_random = cfg.INPUT.LIDAR.FOG.FRACTION_RANDOM

        self.delta_l = cfg.INPUT.LIDAR.PROJECTION.DELTA_L
        self.pixel_l = cfg.INPUT.LIDAR.PROJECTION.PIXEL_L
        self.h1 = cfg.INPUT.LIDAR.PROJECTION.HEIGHT_LB
        self.h2 = cfg.INPUT.LIDAR.PROJECTION.HEIGHT_UB
        self.delta_h = cfg.INPUT.LIDAR.PROJECTION.DELTA_H
        
        self.mode = mode

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        dataset_dict['height'] = self.pixel_l
        dataset_dict['width'] = self.pixel_l
        data_shape = np.array([self.pixel_l, self.pixel_l])

        data_root = dataset_dict['data_root']
        timestamp = dataset_dict['timestamp']

        radar_path = os.path.join(data_root, 'radar')
        radar_intensity_name = os.path.join(radar_path, timestamp+'.jpg')
        radar_intensity = utils.read_image(radar_intensity_name, format='L')
        assert self.pixel_l == radar_intensity.shape[0], 'radar image width does not match INPUT.LIDAR.PROJECTION.PIXEL_L in the config file.'
        assert self.pixel_l == radar_intensity.shape[1], 'radar image height does not match INPUT.LIDAR.PROJECTION.PIXEL_L in the config file.'
        radar_intensity = radar_intensity / 255.0
        radar_intensity = torch.as_tensor(np.ascontiguousarray(radar_intensity.transpose(2, 0, 1)))
        
        if self.history_on:
            radar_intensity = [radar_intensity]
            radar_history_path = os.path.join(data_root, 'radar_history')
            for i in range(1, self.num_history+1):
                radar_history_name = os.path.join(radar_history_path, timestamp+'_'+str(i)+'.jpg')
                radar_history = utils.read_image(radar_history_name, format='L')
                radar_history = radar_history / 255.0
                radar_intensity.append(torch.as_tensor(np.ascontiguousarray(radar_history.transpose(2, 0, 1))))

        dataset_dict['radar_intensity'] = radar_intensity

        clear_lidar = random.uniform(0,1) >= self.fog_ratio
        train_beta = random.uniform(self.beta_range[0], self.beta_range[1])

        if self.mode == 'train':
            lidar_path = os.path.join(data_root, 'lidar')
            lidar_name = os.path.join(lidar_path, timestamp+'.bin')
            lidar_data = np.fromfile(lidar_name, dtype=np.float32)
            lidar_data = lidar_data.reshape((-1, 4))
            if not clear_lidar:
                lidar_data, _ = lidar_foggify(lidar_data, train_beta)
        else:
            if self.eval_beta == 0:
                lidar_path = os.path.join(data_root, 'lidar')
            else:
                lidar_path = os.path.join(data_root, ('lidar_fog_'+str(self.eval_beta)).rstrip('0'))
            lidar_name = os.path.join(lidar_path, timestamp+'.bin')
            lidar_data = np.fromfile(lidar_name, dtype=np.float32)
            lidar_data = lidar_data.reshape((-1, 4))
        lidar_intensity, lidar_occupancy = lidar_pc2pixor(
            lidar_data, self.delta_l, self.pixel_l, self.h1, self.h2, self.delta_h)
        lidar_intensity = torch.as_tensor(np.ascontiguousarray(np.expand_dims(lidar_intensity, -1).transpose(2, 0, 1)))
        lidar_occupancy = torch.as_tensor(np.ascontiguousarray(lidar_occupancy.transpose(2, 0, 1)))

        if self.history_on:
            lidar_intensity = [lidar_intensity]
            lidar_occupancy = [lidar_occupancy]

            for i in range(1, self.num_history+1):
                if self.mode == "train":
                    lidar_history_path = os.path.join(data_root, "lidar_history")
                    lidar_history_name = os.path.join(lidar_history_path, timestamp+'_'+str(i)+'.bin')
                    lidar_history_data = np.fromfile(lidar_history_name, dtype=np.float32)
                    lidar_history_data = lidar_history_data.reshape((-1, 4))
                    lidar_history_T_name = os.path.join(lidar_history_path, timestamp+'_'+str(i)+'_T.bin')
                    lidar_history_T = np.fromfile(lidar_history_T_name, dtype=np.float32)
                    lidar_history_T = np.matrix(lidar_history_T.reshape((4, 4))).T
                    if not clear_lidar:
                        lidar_history_data, _ = lidar_foggify(lidar_history_data,train_beta)
                else:
                    if self.eval_beta == 0:
                        lidar_history_path = os.path.join(data_root, "lidar_history")
                    else:
                        lidar_history_path = os.path.join(data_root, ("lidar_history_fog_"+str(self.eval_beta)).rstrip('0'))
                    lidar_history_name = os.path.join(lidar_history_path, timestamp+'_'+str(i)+'.bin')
                    lidar_history_data = np.fromfile(lidar_history_name, dtype=np.float32)
                    lidar_history_data = lidar_history_data.reshape((-1, 4))
                    lidar_history_T_name = os.path.join(lidar_history_path, timestamp+'_'+str(i)+'_T.bin')
                    lidar_history_T = np.fromfile(lidar_history_T_name, dtype=np.float32)
                    lidar_history_T = np.matrix(lidar_history_T.reshape((4, 4))).T

                lidar_history_homo = np.concatenate([lidar_history_data[:,0:3], np.ones([lidar_history_data.shape[0], 1])], axis=1)
                lidar_history_homo = np.matrix(lidar_history_homo).T
                lidar_history_homo = np.array((lidar_history_T * lidar_history_homo).T)
                lidar_history_data[:,0:3] = lidar_history_homo[:,0:3]
                lidar_history_intensity, lidar_history_occupancy = lidar_pc2pixor(
                    lidar_history_data, self.delta_l, self.pixel_l, self.h1, self.h2, self.delta_h)
                lidar_intensity.append(torch.as_tensor(np.ascontiguousarray(np.expand_dims(lidar_history_intensity, -1).transpose(2, 0, 1))))
                lidar_occupancy.append(torch.as_tensor(np.ascontiguousarray(lidar_history_occupancy.transpose(2, 0, 1))))

        dataset_dict["lidar_intensity"] = lidar_intensity
        dataset_dict["lidar_occupancy"] = lidar_occupancy

        annos = dataset_dict["annotations"]
        for car in annos:
            car["bbox"][0] = car["bbox"][0] / self.delta_l + (self.pixel_l - 1) / 2.0
            car["bbox"][1] = car["bbox"][1] / self.delta_l + (self.pixel_l - 1) / 2.0
            car["bbox"][2] = car["bbox"][2] / self.delta_l
            car["bbox"][3] = car["bbox"][3] / self.delta_l
            if car["bbox"][4] >= -112.5 and car["bbox"][4] < 67.5:
                direction = 0
            else:
                direction = 1
                if car["bbox"][4] < -112.5:
                    car["bbox"][4] = car["bbox"][4] + 180
                else:
                    car["bbox"][4] = car["bbox"][4] - 180
            car["direction"] = direction
        instances = annotations_to_instances_directed(annos, data_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict