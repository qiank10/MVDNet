# Tool to prepare foggy lidar test set from Oxford RobotCar dataset.
# Adapted from https://github.com/princeton-computational-imaging/SeeingThroughFog/blob/master/tools/DatasetFoggification/lidar_foggification.py
# Licensed under the Apache License

import argparse, tqdm, os, errno
import numpy as np
from transform import build_se3_transform

class BetaRandomization():

    def __init__(self, beta):
        self.mhf = 2 # maximal horzontal frequency
        self.mvf = 5 # maximal vertical frequency
        self.height_max = 5
        self.offset = []

        self.beta = beta

        # sample number of furier components, sample random offsets to one another, # Independence Height and angle
        self.number_height = np.random.randint(3,5)
        self.number_angle = np.random.randint(6,10)

        # sample frequencies
        self.frequencies_angle = np.random.randint(1, self.mhf, size=self.number_angle)
        self.frequencies_height = np.random.randint(0, self.mvf, size=self.number_angle)
        # sample frequencies
        self.offseta = np.random.uniform(0, 2*np.pi, size=self.number_angle)
        self.offseth = np.random.uniform(0, 2*np.pi, size=self.number_angle)
        self.intensitya = np.random.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)
        self.intensityh = np.random.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)

        pass

    def _function(self, angle_h=None, height=None):
        was_None = False
        if height is None:
            height = np.linspace(0, self.height_max, 200)/self.height_max*2*np.pi
            was_None = True

        if angle_h is None:
            angle_h = np.linspace(0, 2*np.pi, 200)
            was_None = True
        a = 0
        h = 0
        if was_None:
            a, h = np.meshgrid(angle_h, height)
        else:
            a = angle_h
            h = height

        output = np.zeros(np.shape(a))
        for fa, fh, oa, oh, Ah, Aa in zip(self.frequencies_angle, self.frequencies_height, self.offseta, self.offseth, self.intensityh, self.intensitya):
            output += np.abs((Aa*np.sin(fa*a+oa)/fa+Ah*np.sin(fa*a+fh*h+oh)))

        output += self.beta
        return output

    def get_beta(self, distance_forward, right, height):
        distance_forward = np.where(distance_forward == 0, np.ones_like(distance_forward) * 0.0001, distance_forward)
        angle = np.arctan2(right, distance_forward)
        beta_usefull = self._function(angle, height)

        return beta_usefull

def lidar_foggification(pts_3D, beta, n=0.02, g=0.45, dmin=2, fraction_random=0.05):
    Randomized_beta = BetaRandomization(beta)
    
    d = np.sqrt(pts_3D[:,0] * pts_3D[:,0] + pts_3D[:,1] * pts_3D[:,1] + pts_3D[:,2] * pts_3D[:,2])
    detectable_points = np.where(d>dmin)
    d = d[detectable_points]
    pts_3D = pts_3D[detectable_points]

    beta_usefull = Randomized_beta.get_beta(pts_3D[:,0], pts_3D[:, 1], pts_3D[:, 2])
    dmax = -np.divide(np.log(np.divide(n,(pts_3D[:,3] + g))),(2 * beta_usefull))
    dnew = -np.log(1 - 0.5) / (beta_usefull)

    probability_lost = 1 - np.exp(-beta_usefull*dmax)
    lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost

    if Randomized_beta.beta == 0.0:
        return pts_3D, np.ones(len(pts_3D))

    cloud_scatter = np.logical_and(dnew < d, np.logical_not(lost))
    random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))
    idx_stable = np.where(d<dmax)[0]
    old_points = np.zeros((len(idx_stable), 4))
    old_class = np.zeros(len(idx_stable))
    old_points[:,0:] = pts_3D[idx_stable,:]
    old_points[:,3] = old_points[:,3]*np.exp(-beta_usefull[idx_stable]*d[idx_stable])
    old_class[:] = 1

    cloud_scatter_idx = np.where(np.logical_and(dmax<d, cloud_scatter))[0]
    cloud_scatter = np.zeros((len(cloud_scatter_idx), 4))
    cloud_class = np.zeros(len(cloud_scatter_idx))
    cloud_scatter[:,0:] =  pts_3D[cloud_scatter_idx,:]
    cloud_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(cloud_scatter[:,0:3]), np.transpose(np.divide(dnew[cloud_scatter_idx],d[cloud_scatter_idx]))))
    cloud_scatter[:,3] = cloud_scatter[:,3]*np.exp(-beta_usefull[cloud_scatter_idx]*dnew[cloud_scatter_idx])
    cloud_class[:] = 0

    # Subsample random scatter abhaengig vom noise im Lidar
    random_scatter_idx = np.where(random_scatter)[0]
    scatter_max = np.min(np.vstack((dmax, d)).transpose(), axis=1)
    drand = np.random.uniform(high=scatter_max[random_scatter_idx])
    # scatter outside min detection range and do some subsampling. Not all points are randomly scattered.
    # Fraction of 0.05 is found empirically.
    drand_idx = np.where(drand>dmin)
    drand = drand[drand_idx]
    random_scatter_idx = random_scatter_idx[drand_idx]
    # Subsample random scattered points to 0.05%
    subsampled_idx = np.random.choice(len(random_scatter_idx), int(fraction_random*len(random_scatter_idx)), replace=False)
    drand = drand[subsampled_idx]
    random_scatter_idx = random_scatter_idx[subsampled_idx]


    random_scatter = np.zeros((len(random_scatter_idx), 4))
    random_class = np.zeros(len(random_scatter_idx))
    random_scatter[:,0:] = pts_3D[random_scatter_idx,:]
    random_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(random_scatter[:,0:3]), np.transpose(drand/d[random_scatter_idx])))
    random_scatter[:,3] = random_scatter[:,3]*np.exp(-beta_usefull[random_scatter_idx]*drand)
    random_class[:] = 0
    
    dist_pts_3d = np.concatenate((old_points, cloud_scatter, random_scatter), axis=0)
    dist_pts_class = np.concatenate((old_class, cloud_class, random_class), axis=0)

    return dist_pts_3d, dist_pts_class

LIDAR_FOG_PATH = None
LIDAR_HISTORY_FOG_PATH = None
LIDAR_PATH = None

def main(args):
    radar_timestamp_path = os.path.join(args.data_path, 'radar.timestamps')
    radar_timestamp = np.loadtxt(radar_timestamp_path)[:,0]

    radar_odometry_path = os.path.join(args.data_path, 'gt/radar_odometry.csv')
    radar_odometry = np.genfromtxt(radar_odometry_path, delimiter=',')[1:,np.r_[9,8,2:8]]

    num_history = 4
    num_frame = len(radar_timestamp)-1

    for i in tqdm.tqdm(range(len(radar_timestamp) - 1)):
        lidar_filename = os.path.join(LIDAR_PATH, str(int(radar_timestamp[i])) + '.bin')
        lidar_data = np.fromfile(lidar_filename, dtype=np.float32).reshape((-1,4))
        lidar_fog_data, _ = lidar_foggification(lidar_data, args.beta)

        lidar_save_path = os.path.join(LIDAR_FOG_PATH, str(int(radar_timestamp[i])) + '.bin')
        lidar_fog_data.astype(np.float32).tofile(lidar_save_path)

        if i >= num_history:
            invT = np.eye(4)
            for j in range(1, num_history+1):
                frameXYZRPY = np.array([radar_odometry[i-j,2], radar_odometry[i-j,3], 0, 0, 0, radar_odometry[i-j,7]])
                frameT = build_se3_transform(frameXYZRPY)
                invT = frameT * invT
                T = np.linalg.inv(invT).T
                lidar_T_path = os.path.join(LIDAR_HISTORY_FOG_PATH, str(int(radar_timestamp[i])) + '_' + str(j) + '_T.bin')
                T.astype(np.float32).tofile(lidar_T_path)
                lidar_history_dst_path = os.path.join(LIDAR_HISTORY_FOG_PATH, str(int(radar_timestamp[i])) + '_' + str(j) + '.bin')
                lidar_history_src_path = os.path.join(LIDAR_FOG_PATH, str(int(radar_timestamp[i-j])) + '.bin')
                try:
                    os.symlink(lidar_history_src_path, lidar_history_dst_path)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        os.remove(lidar_history_dst_path)
                        os.symlink(lidar_history_src_path, lidar_history_dst_path)
                    else:
                        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare foggy lidar test set from Oxford RobotCar radar dataset')
    parser.add_argument('--data_path', type=str, required=True, help='path to the data record folder')
    parser.add_argument('--beta', type=float, required=True, help='fog density (0.005 - 0.08)')

    args = parser.parse_args()

    processed_path = os.path.join(args.data_path, 'processed')
    if not os.path.isdir(processed_path):
        os.mkdir(processed_path)

    LIDAR_FOG_PATH = os.path.join(processed_path, 'lidar_fog_' + str(args.beta))
    if not os.path.isdir(LIDAR_FOG_PATH):
        os.mkdir(LIDAR_FOG_PATH)

    LIDAR_HISTORY_FOG_PATH = os.path.join(processed_path, 'lidar_history_fog_' + str(args.beta))
    if not os.path.isdir(LIDAR_HISTORY_FOG_PATH):
        os.mkdir(LIDAR_HISTORY_FOG_PATH)

    LIDAR_PATH = os.path.join(args.data_path, 'processed', 'lidar')
    assert os.path.isdir(LIDAR_PATH), "Lidar data is missing, please generate lidar data first!"

    main(args)