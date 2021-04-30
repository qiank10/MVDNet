import copy
import itertools
import logging
import numpy as np
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import Params
from tabulate import tabulate
from collections import defaultdict
import pickle
import json

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.evaluation.rotated_coco_evaluation import RotatedCOCOeval
from detectron2.utils.logger import create_small_table
from detectron2.structures import RotatedBoxes, pairwise_iou_rotated
from mvdnet.data.datasets import get_robotcar_dicts

logger = logging.getLogger(__name__)

def create_coco_api(robotcar_dataset):
    coco_api = COCO()
    coco_api.dataset = dict()
    coco_api.anns = dict()
    coco_api.cats = dict()
    coco_api.imgs = dict()
    coco_api.imgToAnns = defaultdict(list)
    coco_api.catToImgs = defaultdict(list)

    coco_api.dataset["images"] = []
    for record in robotcar_dataset:
        image = dict()
        image["timestamp"] = record["timestamp"]
        image["id"] = record["image_id"]
        coco_api.dataset["images"].append(image)

    coco_api.dataset["categories"] = []
    category = dict()
    category["supercategory"] = "vehicle"
    category["id"] = 1
    category["name"] = "car"
    coco_api.dataset["categories"].append(category)

    coco_api.dataset["annotations"] = []
    for record in robotcar_dataset:
        for ann in record["annotations"]:
            coco_ann = dict()
            coco_ann["area"] = ann["bbox"][2] * ann["bbox"][3]
            coco_ann["iscrowd"] = ann["iscrowd"]
            coco_ann["image_id"] = record["image_id"]
            coco_ann["bbox"] = copy.deepcopy(ann["bbox"])
            coco_ann["category_id"] = 1
            coco_ann["id"] = ann["id"]
            coco_api.dataset["annotations"].append(coco_ann)

    coco_api.createIndex()
    return coco_api

class RobotCarEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)

        logger.info("Loading unique objects from {}...".format(dataset_name))

        data_root = self._metadata.data_root
        split_path = self._metadata.split_path
        pixel_l = cfg.INPUT.LIDAR.PROJECTION.PIXEL_L
        delta_l = cfg.INPUT.LIDAR.PROJECTION.DELTA_L
        robotcar_dataset = get_robotcar_dicts(data_root, split_path)
        for record in robotcar_dataset:
            for car in record["annotations"]:
                car["bbox"][0] = car["bbox"][0] / delta_l + (pixel_l - 1) / 2.0
                car["bbox"][1] = car["bbox"][1] / delta_l + (pixel_l - 1) / 2.0
                car["bbox"][2] = car["bbox"][2] / delta_l
                car["bbox"][3] = car["bbox"][3] / delta_l
        self._coco_api = create_coco_api(robotcar_dataset)

        logger.info("Unique objects loaded: {}".format(len(robotcar_dataset)))

        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        return ("bbox",)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = self.instances_to_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            self._predictions.append(prediction)

    def instances_to_json(self, instances, img_id):
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }

            results.append(result)
        return results

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[RobotcarEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)

        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            assert task == "bbox", "Task {} is not supported".format(task)
            coco_eval = (
                self._evaluate_predictions_on_coco(self._coco_api, coco_results)
                if len(coco_results) > 0
                else None
            )
            if self._output_dir:
                coco_eval.eval["precision"][:,:,0,0,2].astype(np.float32).tofile(os.path.join(self._output_dir, "precision.bin"))
            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _evaluate_predictions_on_coco(self, coco_gt, coco_results):
        assert len(coco_results) > 0

        coco_dt = coco_gt.loadRes(coco_results)
        for record in coco_gt.dataset["annotations"]:
            record["bbox"] = [round(val, 2) for val in record["bbox"]]
        for record in coco_dt.dataset["annotations"]:
            record["bbox"] = [round(val, 2) for val in record["bbox"]]

        coco_eval = RobotCarCOCOeval(coco_gt, coco_dt, iouType="bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
            "keypoints": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results

        precisions = coco_eval.eval["precision"]
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def _eval_box_proposals(self, predictions):
        if self._output_dir:
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        stats = _evaluate_box_proposals(predictions, self._coco_api, area="all", limit=1000)
        key = "AP@1000"
        res[key] = float(stats["ap"].item() * 100)
        key = "AR@1000"
        res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    prop_overlaps = []
    num_pos = 0
    num_prop = 0
    prop_cnt = {}
    gt_cnt = {}
    gt_iou = {}

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            obj["bbox"]
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 5)  # guard against no boxes
        gt_boxes = RotatedBoxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        # print(len(predictions))

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)
        num_prop += len(predictions.proposal_boxes)

        assert len(predictions.proposal_boxes) <= limit

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou_rotated(predictions.proposal_boxes, gt_boxes)
        _prop_overlaps, _prop_ind = overlaps.max(dim=1)

        assert len(predictions) == len(_prop_ind)

        from collections import Counter
        _prop_ind = Counter(_prop_ind[_prop_overlaps >= 0.5].cpu().detach().numpy())
        _prop_cnt = np.zeros(len(gt_boxes))
        for ind, cnt in _prop_ind.items():
            _prop_cnt[ind] = cnt
        prop_cnt["x%d" % prediction_dict["image_id"]] = _prop_cnt

        _gt_overlaps, _gt_ind = overlaps.max(dim=0)
        _gt_ind = ((_gt_overlaps >= 0.7) + 0).cpu().detach().numpy()
        gt_cnt["x%d" % prediction_dict["image_id"]] = _gt_ind
        gt_iou["x%d" % prediction_dict["image_id"]] = _gt_overlaps.cpu().detach().numpy()

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
        prop_overlaps.append(_prop_overlaps)

    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)
    prop_overlaps = (
        torch.cat(prop_overlaps, dim=0) if len(prop_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    prop_overlaps, _ = torch.sort(prop_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.7, 0.7 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    precisions = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
        precisions[i] = (prop_overlaps >= t).float().sum() / float(num_prop)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    ap = precisions.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "ap": ap,
        "precisions": precisions,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
        "prop_cnt": prop_cnt,
        "gt_cnt": gt_cnt,
        "gt_iou": gt_iou
    }

class RobotCarCOCOeval(RotatedCOCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox'):
        if not iouType:
            print('iouType not specified. use default iouType segm')

        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = RobotCarParams(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def summarize(self):
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.65, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=.8, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(0, iouThr=.65, maxDets=self.params.maxDets[0])
            stats[5] = _summarize(0, iouThr=.65, maxDets=self.params.maxDets[1])
            stats[6] = _summarize(0, iouThr=.65, maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        assert iouType == 'bbox', 'only bbox evaluation is supported for RobotCar dataset!'
        summarize = _summarizeDets
        self.stats = summarize()

class RobotCarParams(Params):
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1