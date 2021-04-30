from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser, default_setup
from detectron2.evaluation import inference_on_dataset

import mvdnet.modeling
from mvdnet.data import RobotCarMapper
from mvdnet.evaluation import RobotCarEvaluator
from mvdnet.config import get_mvdnet_cfg_defaults

def setup(args):
    cfg = get_cfg()
    get_mvdnet_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=RobotCarMapper(cfg, "eval"))

    evaluator = RobotCarEvaluator(dataset_name, cfg, True, cfg.OUTPUT_DIR)

    result = inference_on_dataset(model, data_loader, evaluator)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)