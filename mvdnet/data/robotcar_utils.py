import numpy as np
import torch
from detectron2.structures import RotatedBoxes, Instances
from detectron2.data import detection_utils as utils

def annotations_to_instances_directed(annos, image_size):
    boxes = [obj['bbox'] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj['category_id'] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    directions = [obj['direction'] for obj in annos]
    directions = torch.tensor(directions, dtype=torch.int64)
    target.gt_directions = directions

    ids = [obj['car_id'] for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    return target

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

def lidar_foggify(pts_3D, beta, n=0.02, g=0.45, dmin=2, fraction_random=0.05):
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

def lidar_pc2pixor(lidar_data, delta_l=0.2, pixel_l=320, h1=-1.0, h2=2.5, delta_h=0.1):
    l1 = (-pixel_l/2) * delta_l
    l2 = (pixel_l/2) * delta_l
    pixel_h = np.int(np.round((h2 - h1) / delta_h))

    lidar_data[:,0] = -lidar_data[:,0]
    idx_x = np.logical_and(lidar_data[:,0] >= l1, lidar_data[:,0] < l2)
    idx_y = np.logical_and(lidar_data[:,1] >= l1, lidar_data[:,1] < l2)
    idx_z = np.logical_and(lidar_data[:,2] >= h1, lidar_data[:,2] < h2)
    idx_valid = np.logical_and(idx_z, np.logical_and(idx_y, idx_x))
    lidar_data = lidar_data[idx_valid, :]

    lidar_bev_idx = np.zeros([len(lidar_data), 2])
    lidar_bev_idx[:,0] = np.floor((lidar_data[:,0] - l1) / delta_l)
    lidar_bev_idx[:,1] = np.floor((lidar_data[:,1] - l1) / delta_l)
    lidar_bev_idx[lidar_bev_idx == pixel_l] = pixel_l - 1
    lidar_bev_idx = lidar_bev_idx.astype(np.int)

    lidar_height_idx = np.floor((lidar_data[:,2] - h1) / delta_h)
    lidar_height_idx[lidar_height_idx == pixel_h] = pixel_h - 1
    lidar_height_idx = lidar_height_idx.astype(np.int)

    lidar_intensity = np.zeros([pixel_l, pixel_l])
    lidar_occupancy = np.zeros([pixel_l, pixel_l, pixel_h])
    for i in range(len(lidar_bev_idx)):
        lidar_occupancy[lidar_bev_idx[i,0], lidar_bev_idx[i,1], lidar_height_idx[i]] = 1
        lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]] = max(lidar_data[i,3], \
            lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]])

    return lidar_intensity, lidar_occupancy