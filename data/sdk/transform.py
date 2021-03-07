# Adapted from https://github.com/ori-mrg/robotcar-dataset-sdk/blob/master/python/transform.py
# Licensed under the Apache License

import numpy as np
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx

def se3_transform(xyzrpy):
    pos = np.array(xyzrpy[0:3])
    rot = R.from_euler('xyz', xyzrpy[3:])
    return rot, pos

def inverse_transform(rot, pos):
    rot_inv = rot.inv()
    pos_inv = np.array((-np.matrix(rot_inv.as_dcm()) * np.matrix(pos).T).T)[0]
    return rot_inv, pos_inv

def compose_transform(rot1, pos1, rot2, pos2):
    rot_com = rot2.__mul__(rot1)
    pos_com = np.array((np.matrix(rot2.as_dcm()) * np.matrix(pos1).T + np.matrix(pos2).T).T)[0]
    return rot_com, pos_com

def frame_transform(src_ts, dst_ts, vo):
    if src_ts == dst_ts:
        return R.from_quat([0,0,0,1]), np.zeros(3)

    if src_ts > dst_ts:
        start_ts = dst_ts
        end_ts = src_ts
    else:
        start_ts = src_ts
        end_ts = dst_ts

    transform_list = np.where(np.logical_and(vo['timestamp'] > start_ts, vo['timestamp'] < end_ts))[0]

    if len(transform_list) > 0:
        rot_s02 = vo['rot'][transform_list[0]-1]
        pos_s02 = vo['pos'][transform_list[0]-1]
        rot_sint = R.from_quat([[0,0,0,1], rot_s02.as_quat()])
        ts_sint = [vo['timestamp'][transform_list[0]-1],vo['timestamp'][transform_list[0]]]
        slerp_s = Slerp(ts_sint, rot_sint)
        rot_s01 = slerp_s([start_ts])
        rot_s01 = R.from_quat(rot_s01.as_quat()[0])
        pos_s01 = (start_ts - ts_sint[0]) / (ts_sint[1] - ts_sint[0]) * pos_s02
        rot_s10, pos_s10 = inverse_transform(rot_s01, pos_s01)
        rot_s12, pos_s12 = compose_transform(rot_s02, pos_s02, rot_s10, pos_s10)

        rot_e02 = vo['rot'][transform_list[-1]]
        pos_e02 = vo['pos'][transform_list[-1]]
        rot_eint = R.from_quat([[0,0,0,1], rot_e02.as_quat()])
        ts_eint = [vo['timestamp'][transform_list[-1]],vo['timestamp'][transform_list[-1]+1]]
        slerp_e = Slerp(ts_eint, rot_eint)
        rot_e01 = slerp_e([end_ts])
        rot_e01 = R.from_quat(rot_e01.as_quat()[0])
        pos_e01 = (end_ts - ts_eint[0]) / (ts_eint[1] - ts_eint[0]) * pos_e02

        rot_se = rot_s12
        pos_se = pos_s12
        for ii in transform_list[:-1]:
            curr_rot = vo['rot'][ii]
            curr_pos = vo['pos'][ii]
            rot_se, pos_se = compose_transform(curr_rot, curr_pos, rot_se, pos_se)
        rot_se, pos_se = compose_transform(rot_e01, pos_e01, rot_se, pos_se)
    else:
        transform_list = np.where(vo['timestamp'] > start_ts)[0]
        transform_idx = transform_list[0] - 1
        rot_int = R.from_quat([[0,0,0,1], vo['rot'][transform_idx].as_quat()])
        ts_int = [vo['timestamp'][transform_idx], vo['timestamp'][transform_idx+1]]
        slerp = Slerp(ts_int, rot_int)
        rot_0se = slerp([start_ts, end_ts])
        rot_0s = R.from_quat(rot_0se.as_quat()[0])
        rot_0e = R.from_quat(rot_0se.as_quat()[1])
        pos_0s = (start_ts - ts_int[0]) / (ts_int[1] - ts_int[0]) * vo['pos'][transform_idx]
        pos_0e = (end_ts - ts_int[0]) / (ts_int[1] - ts_int[0]) * vo['pos'][transform_idx]
        rot_s0, pos_s0 = inverse_transform(rot_0s, pos_0s)
        rot_se, pos_se = compose_transform(rot_0e, pos_0e, rot_s0, pos_s0)

    if src_ts > dst_ts:
        return rot_se, pos_se
    else:
        rot_es, pos_es = inverse_transform(rot_se, pos_se)
        return rot_es, pos_es