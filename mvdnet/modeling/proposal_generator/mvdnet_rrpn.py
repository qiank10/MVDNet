from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn
import itertools

from detectron2.layers import ShapeSpec, batched_nms_rotated, cat
from detectron2.structures import Instances, RotatedBoxes
from detectron2.utils.registry import Registry

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransformRotated
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rrpn_outputs import RRPNOutputs
from detectron2.modeling.proposal_generator.rpn import build_rpn_head

def find_top_final_fusion_rrpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
    in_features,
):
    image_sizes = images.image_sizes  # in (h, w) order
    num_images = len(image_sizes)
    device = proposals[in_features[0]][0].device
    pre_results = {}

    for f in in_features:
        topk_scores = []  # #lvl Tensor, each of shape N x topk
        topk_proposals = []
        level_ids = []  # #lvl Tensor, each of shape (topk,)
        batch_idx = torch.arange(num_images, device=device)
        for level_id, proposals_i, logits_i in zip(
            itertools.count(), proposals[f], pred_objectness_logits[f]
        ):
            Hi_Wi_A = logits_i.shape[1]
            num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

            # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
            # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
            logits_i, idx = logits_i.sort(descending=True, dim=1)
            topk_scores_i = logits_i[batch_idx, :num_proposals_i]
            topk_idx = idx[batch_idx, :num_proposals_i]

            # each is N x topk
            topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 5

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

        # 2. Concat all levels together
        topk_scores = cat(topk_scores, dim=1)
        topk_proposals = cat(topk_proposals, dim=1)
        level_ids = cat(level_ids, dim=0)

        # 3. For each image, run a per-level NMS, and choose topk results.
        pre_results[f] = []
        for n, image_size in enumerate(image_sizes):
            boxes = RotatedBoxes(topk_proposals[n])
            scores_per_img = topk_scores[n]
            valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
            if not valid_mask.all():
                boxes = boxes[valid_mask]
                scores_per_img = scores_per_img[valid_mask]
            boxes.clip(image_size)

            # filter empty boxes
            keep = boxes.nonempty(threshold=min_box_side_len)
            lvl = level_ids
            if keep.sum().item() != len(boxes):
                boxes, scores_per_img, lvl = (boxes[keep], scores_per_img[keep], level_ids[keep])

            keep = batched_nms_rotated(boxes.tensor, scores_per_img, lvl, nms_thresh)
            # In Detectron1, there was different behavior during training vs. testing.
            # (https://github.com/facebookresearch/Detectron/issues/459)
            # During training, topk is over the proposals from *all* images in the training batch.
            # During testing, it is over the proposals for each image separately.
            # As a result, the training behavior becomes batch-dependent,
            # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
            # This bug is addressed in Detectron2 to make the behavior independent of batch size.
            keep = keep[:post_nms_topk]

            res = Instances(image_size)
            res.proposal_boxes = boxes[keep]
            res.objectness_logits = scores_per_img[keep]
            res.level_ids = lvl[keep]
            pre_results[f].append(res)

    # 4. For each image, run a per-level NMS across all feature maps.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = RotatedBoxes.cat([pre_results[f][n].proposal_boxes for f in in_features])
        scores_per_img = cat([pre_results[f][n].objectness_logits for f in in_features], dim=0)
        lvl = cat([pre_results[f][n].level_ids for f in in_features], dim=0)

        keep = batched_nms_rotated(boxes.tensor, scores_per_img, lvl, nms_thresh)

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    
    return results

@PROPOSAL_GENERATOR_REGISTRY.register()
class MVDNetRRPN(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT

        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = {}
        for f in self.in_features:
            self.anchor_generator[f] = build_anchor_generator(
                cfg, [input_shape[f]]
            )
            self.add_module("%s_anchor_generator"%f, self.anchor_generator[f])
        self.box2box_transform = Box2BoxTransformRotated(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )

        self.rpn_head = {}
        for f in self.in_features:
            self.rpn_head[f] = build_rpn_head(cfg, [input_shape[f]])
            self.add_module("%s_rpn_head"%f, self.rpn_head[f])

    def forward(self, images, features, gt_instances=None):
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        pred_objectness_logits = {}
        pred_anchor_deltas = {}
        anchors = {}
        for f in self.in_features:
            pred_objectness_logits[f], pred_anchor_deltas[f] = self.rpn_head[f]([features[f]])
            anchors[f] = self.anchor_generator[f]([features[f]])

        outputs = {}
        for f in self.in_features:
            outputs[f] = RRPNOutputs(
                self.box2box_transform,
                self.anchor_matcher,
                self.batch_size_per_image,
                self.positive_fraction,
                images,
                pred_objectness_logits[f],
                pred_anchor_deltas[f],
                anchors[f],
                self.boundary_threshold,
                gt_boxes,
                self.smooth_l1_beta,
            )

        losses = {}
        if self.training:
            for f in self.in_features:
                for loss_key, loss_val in outputs[f].losses().items():
                    losses["%s_%s"%(f,loss_key)] = loss_val

        predict_proposals = {}
        predict_objectness_logits = {}
        for f in self.in_features:
            predict_proposals[f] = outputs[f].predict_proposals()
            predict_objectness_logits[f] = outputs[f].predict_objectness_logits()
        proposals = find_top_final_fusion_rrpn_proposals(
            predict_proposals,
            predict_objectness_logits,
            images,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_side_len,
            self.training,
            self.in_features
        )

        return proposals, losses