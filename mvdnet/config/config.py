from detectron2.config import CfgNode as CN

def get_mvdnet_cfg_defaults(cfg):

    cfg.INPUT.EVAL_BETA = 0.05
    cfg.INPUT.HISTORY_ON = True
    cfg.INPUT.NUM_HISTORY = 4    

    cfg.INPUT.LIDAR = CN()
    cfg.INPUT.LIDAR.FOG = CN()
    cfg.INPUT.LIDAR.FOG.N = 0.02
    cfg.INPUT.LIDAR.FOG.G = 0.45
    cfg.INPUT.LIDAR.FOG.D_MIN = 2
    cfg.INPUT.LIDAR.FOG.FRACTION_RANDOM=0.05
    cfg.INPUT.LIDAR.FOG.FOG_RATIO = 0.5
    cfg.INPUT.LIDAR.FOG.BETA_RANGE = [0.005, 0.08]

    cfg.INPUT.LIDAR.PROJECTION = CN()
    cfg.INPUT.LIDAR.PROJECTION.DELTA_L = 0.2
    cfg.INPUT.LIDAR.PROJECTION.PIXEL_L = 320
    cfg.INPUT.LIDAR.PROJECTION.HEIGHT_LB = -1.0
    cfg.INPUT.LIDAR.PROJECTION.HEIGHT_UB = 2.5
    cfg.INPUT.LIDAR.PROJECTION.DELTA_H = 0.1

    cfg.MODEL.MVDNET = CN()
    cfg.MODEL.MVDNET.STEM_OUT_CHANNELS = 32
    cfg.MODEL.MVDNET.STAGE_LAYERS = [3,3]
    cfg.MODEL.MVDNET.RADAR_STAGE_OUT_CHANNELS = [32, 64]
    cfg.MODEL.MVDNET.LIDAR_STAGE_OUT_CHANNELS = [64, 128]
    cfg.MODEL.MVDNET.RADAR_NORM = "BN"
    cfg.MODEL.MVDNET.LIDAR_NORM = "BN"