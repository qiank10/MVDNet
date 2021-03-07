# Tool to prepare radar data from Oxford RobotCar dataset.
# Licensed under the Apache License

import argparse, tqdm, os
import numpy as np
from transform import build_se3_transform
from radar import load_radar, radar_polar_to_cartesian
import cv2

RADAR_PATH = None
RADAR_HISTORY_PATH = None

def main(args):
    timestamp_path = os.path.join(args.data_path, 'radar.timestamps')
    timestamp = np.loadtxt(timestamp_path)[:,0]

    radar_odometry_path = os.path.join(args.data_path, 'gt/radar_odometry.csv')
    radar_odometry = np.genfromtxt(radar_odometry_path, delimiter=',')[1:,np.r_[9,8,2:8]]

    num_history = 4
    num_frame = min(radar_odometry.shape[0], len(timestamp)-1)
    for i in tqdm.tqdm(range(num_history,num_frame)):
        invT = np.eye(4)
        for j in range(0,num_history+1):
            # Calculate tranform matrix
            if j > 0:
                frameXYZRPY = np.array([radar_odometry[i-j,2], radar_odometry[i-j,3], 0, 0, 0, radar_odometry[i-j,7]])
                frameT = build_se3_transform(frameXYZRPY)
                invT = frameT * invT
            T = np.linalg.inv(invT)[np.ix_([0,1,3], [0,1,3])]
            x0 = T[0,-1]
            y0 = T[1,-1]
            yaw0 = np.arctan2(T[1,0], T[0,0])
            x1 = radar_odometry[i-j,2]
            y1 = radar_odometry[i-j,3]
            yaw1 = radar_odometry[i-j,7]

            # Load raw polar radar image
            filename = os.path.join(args.data_path, 'radar', str(int(timestamp[i-j])) + '.png')

            if not os.path.isfile(filename):
                raise FileNotFoundError("Could not find radar example: {}".format(filename))

            fine_timestamp, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
            fine_timestamp = fine_timestamp.T[0]
            fine_timestamp = (fine_timestamp - timestamp[i-j]) / (fine_timestamp[-1] - timestamp[i-j])

            # Process polar radar image
            radar_image = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, args.resolution, args.image_size, 
                True, fine_timestamp, x0, y0, yaw0, x1, y1, yaw1)
            radar_image = (radar_image * 255).astype(int)

            # Save caterisan radar image
            if j == 0:
                save_path = os.path.join(RADAR_PATH, str(int(timestamp[i])) + '.jpg')
            else:
                save_path = os.path.join(RADAR_HISTORY_PATH, str(int(timestamp[i])) + '_' + str(j) + '.jpg')
            cv2.imwrite(save_path, radar_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare radar data from Oxford RobotCar radar dataset')
    parser.add_argument('--data_path', type=str, required=True, help='path to the data record folder')
    parser.add_argument('--image_size', type=int, default=320, help='cartesian image size (px)')
    parser.add_argument('--resolution', type=float, default=0.2, help='cartesian image resultion (m)')

    args = parser.parse_args()

    processed_path = os.path.join(args.data_path, 'processed')
    if not os.path.isdir(processed_path):
        os.mkdir(processed_path)

    RADAR_PATH = os.path.join(processed_path, 'radar')
    if not os.path.isdir(RADAR_PATH):
        os.mkdir(RADAR_PATH)

    RADAR_HISTORY_PATH = os.path.join(processed_path, 'radar_history')
    if not os.path.isdir(RADAR_HISTORY_PATH):
        os.mkdir(RADAR_HISTORY_PATH)

    main(args)