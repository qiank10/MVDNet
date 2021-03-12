# Tool to prepare lidar data from Oxford RobotCar dataset.
# Licensed under the Apache License

import argparse, tqdm, os, errno
import numpy as np
from transform import (
    build_se3_transform, 
    se3_transform, 
    inverse_transform,
    compose_transform,
    frame_transform
)
from radar import load_radar, radar_polar_to_cartesian
import cv2

LIDAR_PATH = None
LIDAR_HISTORY_PATH = None
RADAR_EXTRINSICS = [-0.71813, 0.12, -0.54479, 0, 0.05, 0]
LEFT_LIDAR_EXTRINSICS = [-0.60072, -0.34077, -0.26837, -0.0053948, -0.041998, -3.1337]
RIGHT_LIDAR_EXTRINSICS = [-0.61153, 0.55676, -0.27023, 0.0027052, -0.041999, -3.1357]
RADAR_ROT, RADAR_POS = se3_transform(RADAR_EXTRINSICS)
RADAR_INV_ROT, RADAR_INV_POS = inverse_transform(RADAR_ROT, RADAR_POS)
LEFT_LIDAR_ROT, LEFT_LIDAR_POS = se3_transform(LEFT_LIDAR_EXTRINSICS)
LEFT_LIDAR_INV_ROT, LEFT_LIDAR_INV_POS = inverse_transform(LEFT_LIDAR_ROT, LEFT_LIDAR_POS)
RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS = se3_transform(RIGHT_LIDAR_EXTRINSICS)
RIGHT_LIDAR_INV_ROT, RIGHT_LIDAR_INV_POS = inverse_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS)

def main(args):
    lidar_odometry_path = os.path.join(args.data_path, 'vo/vo.csv')
    lidar_odometry = np.genfromtxt(lidar_odometry_path, delimiter=',')[1:]
    lidar_odometry_rot = []
    lidar_odometry_pos = []
    lidar_odometry_timestamp = []
    for sample in lidar_odometry:
        sample_rot, sample_pos = se3_transform(sample[2:])
        lidar_odometry_rot.append(sample_rot)
        lidar_odometry_pos.append(sample_pos)
        lidar_odometry_timestamp.append(int(sample[1]))
    lidar_odometry_timestamp.append(sample[0])
    lidar_odometry = {'rot':lidar_odometry_rot, 
                    'pos':lidar_odometry_pos,
                    'timestamp':lidar_odometry_timestamp}

    radar_timestamp_path = os.path.join(args.data_path, 'radar.timestamps')
    radar_timestamp = np.loadtxt(radar_timestamp_path)[:,0]

    left_lidar_timestamp_path = os.path.join(args.data_path, 'velodyne_left.timestamps')
    left_lidar_timestamp = np.loadtxt(left_lidar_timestamp_path)[:,0]

    right_lidar_timestamp_path = os.path.join(args.data_path, 'velodyne_right.timestamps')
    right_lidar_timestamp = np.loadtxt(right_lidar_timestamp_path)[:,0]

    radar_odometry_path = os.path.join(args.data_path, 'gt/radar_odometry.csv')
    radar_odometry = np.genfromtxt(radar_odometry_path, delimiter=',')[1:,np.r_[9,8,2:8]]

    num_history = 4
    num_frame = len(radar_timestamp)-1
    lidar_radar_frame_ratio = 5
    right_lidar_timestamp_i = 0
    for radar_timestamp_i in tqdm.tqdm(range(len(radar_timestamp) - 1)):
        while right_lidar_timestamp[right_lidar_timestamp_i] <= radar_timestamp[radar_timestamp_i]:
            right_lidar_timestamp_i += 1
        lidar_data_all = []
        for right_lidar_timestamp_j in range(right_lidar_timestamp_i-1, right_lidar_timestamp_i+lidar_radar_frame_ratio+1):
            right_lidar_filename = os.path.join(args.data_path, 'velodyne_right', str(int(right_lidar_timestamp[right_lidar_timestamp_j])) + '.bin')
            right_lidar_data = np.fromfile(right_lidar_filename, dtype=np.float32)
            right_lidar_data = np.matrix(np.reshape(right_lidar_data, (4, -1)))
            
            left_lidar_timestamp_closest = left_lidar_timestamp[min(range(len(left_lidar_timestamp)), key=lambda ii: abs(left_lidar_timestamp[ii] - right_lidar_timestamp[right_lidar_timestamp_j]))]
            left_lidar_filename = os.path.join(args.data_path, 'velodyne_left', str(int(left_lidar_timestamp_closest)) + '.bin')
            left_lidar_data = np.fromfile(left_lidar_filename, dtype=np.float32)
            left_lidar_data = np.matrix(np.reshape(left_lidar_data, (4, -1)))

            frame_rot, frame_pos = frame_transform(left_lidar_timestamp_closest, right_lidar_timestamp[right_lidar_timestamp_j], lidar_odometry)
            frame_rot, frame_pos = compose_transform(LEFT_LIDAR_ROT, LEFT_LIDAR_POS, frame_rot, frame_pos)
            frame_rot, frame_pos = compose_transform(frame_rot, frame_pos, RIGHT_LIDAR_INV_ROT, RIGHT_LIDAR_INV_POS)
            left_lidar_data[0:3,:] = np.matrix(frame_rot.as_dcm()) * left_lidar_data[0:3,:] + np.tile(np.matrix(frame_pos).T, (1, left_lidar_data.shape[1]))

            lidar_data = np.concatenate((right_lidar_data, left_lidar_data), axis=1)
            lidar_self = np.logical_and(np.logical_and(lidar_data[0,:] > -2.1, lidar_data[0,:] < 2), np.logical_and(lidar_data[1,:] > -0.5, lidar_data[1,:] < 1.4))
            lidar_sel = np.where(np.logical_not(lidar_self))[1]
            lidar_data = lidar_data[:,lidar_sel]
            lidar_location = lidar_data[0:3,:]
            lidar_intensity = lidar_data[3:,:]
            
            frame_rot, frame_pos = frame_transform(right_lidar_timestamp[right_lidar_timestamp_j], radar_timestamp[radar_timestamp_i], lidar_odometry)
            frame_rot, frame_pos = compose_transform(RIGHT_LIDAR_ROT, RIGHT_LIDAR_POS, frame_rot, frame_pos)
            frame_rot, frame_pos = compose_transform(frame_rot, frame_pos, RADAR_INV_ROT, RADAR_INV_POS)
            frame_rot = np.matrix(frame_rot.as_dcm())
            frame_pos = np.matrix(frame_pos).T
            frame_pos = np.tile(frame_pos, (1, lidar_location.shape[1]))

            lidar_location = np.array((frame_rot * np.matrix(lidar_location) + frame_pos))
            lidar_angle = np.arctan2(lidar_location[1,:], lidar_location[0,:])
            lidar_intensity = lidar_intensity / 255.0

            if right_lidar_timestamp_j == right_lidar_timestamp_i - 1:
                sector_bound = 0
                sector_width = 2*np.pi/(lidar_radar_frame_ratio+1)
            elif right_lidar_timestamp_j == right_lidar_timestamp_i + lidar_radar_frame_ratio:
                sector_bound = lidar_radar_frame_ratio*2*np.pi/(lidar_radar_frame_ratio+1)
                sector_width = 2*np.pi/(lidar_radar_frame_ratio+1)
            else:
                sector_bound = (right_lidar_timestamp_j-right_lidar_timestamp_i)*2*np.pi/(lidar_radar_frame_ratio+1)
                sector_width = 2*2*np.pi/(lidar_radar_frame_ratio+1)

            lidar_angle = np.mod(lidar_angle - sector_bound, 2*np.pi)
            lidar_sel = np.logical_and(lidar_angle >= 0, lidar_angle < sector_width)
            lidar_location = lidar_location[:,lidar_sel]
            lidar_intensity = lidar_intensity[:,lidar_sel]
            lidar_data = np.concatenate((lidar_location, lidar_intensity), axis=0)
            lidar_data_all.append(lidar_data)

        lidar_data_all = np.concatenate(lidar_data_all, axis=1).T
        lidar_data_path = os.path.join(LIDAR_PATH, str(int(radar_timestamp[radar_timestamp_i])) + '.bin')
        lidar_data_all.astype(np.float32).tofile(lidar_data_path)

        if radar_timestamp_i >= num_history:
            invT = np.eye(4)
            for radar_timestamp_j in range(1, num_history+1):
                frameXYZRPY = np.array([radar_odometry[radar_timestamp_i-radar_timestamp_j,2], radar_odometry[radar_timestamp_i-radar_timestamp_j,3], 0, 0, 0, radar_odometry[radar_timestamp_i-radar_timestamp_j,7]])
                frameT = build_se3_transform(frameXYZRPY)
                invT = frameT * invT
                T = np.linalg.inv(invT).T
                lidar_T_path = os.path.join(LIDAR_HISTORY_PATH, str(int(radar_timestamp[radar_timestamp_i])) + '_' + str(radar_timestamp_j) + '_T.bin')
                T.astype(np.float32).tofile(lidar_T_path)
                lidar_history_dst_path = os.path.join(LIDAR_HISTORY_PATH, str(int(radar_timestamp[radar_timestamp_i])) + '_' + str(radar_timestamp_j) + '.bin')
                lidar_history_src_path = os.path.join(LIDAR_PATH, str(int(radar_timestamp[radar_timestamp_i-radar_timestamp_j])) + '.bin')
                try:
                    os.symlink(lidar_history_src_path, lidar_history_dst_path)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        os.remove(lidar_history_dst_path)
                        os.symlink(lidar_history_src_path, lidar_history_dst_path)
                    else:
                        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare lidar data from Oxford RobotCar radar dataset')
    parser.add_argument('--data_path', type=str, required=True, help='path to the data record folder')
    
    args = parser.parse_args()

    processed_path = os.path.join(args.data_path, 'processed')
    if not os.path.isdir(processed_path):
        os.mkdir(processed_path)

    LIDAR_PATH = os.path.join(processed_path, 'lidar')
    if not os.path.isdir(LIDAR_PATH):
        os.mkdir(LIDAR_PATH)

    LIDAR_HISTORY_PATH = os.path.join(processed_path, 'lidar_history')
    if not os.path.isdir(LIDAR_HISTORY_PATH):
        os.mkdir(LIDAR_HISTORY_PATH)

    main(args)