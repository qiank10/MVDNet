import logging
import numpy as np
from typing import Dict
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms, Linear, cat, batched_nms_rotated
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated, BoxMode, ImageList
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransformRotated
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.rotated_fast_rcnn import RROIHeads
from typing import List, Optional, Tuple

def fast_rcnn_inference_direction(
    boxes, scores, direct_scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    result_per_image = [
        fast_rcnn_inference_single_image_direction(
            boxes_per_image, scores_per_image, direct_scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape, direct_scores_per_image in zip(scores, boxes, image_shapes, direct_scores)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_direction(
    boxes, scores, direct_scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1) & torch.isfinite(direct_scores).all(dim=1)
    valid_inds = torch.arange(0, len(boxes), dtype=torch.int64, device=boxes.device)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        direct_scores = direct_scores[valid_mask]
        valid_inds = valid_inds[valid_mask]

    B = 5  # box dimension
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // B
    # Convert to Boxes to use the `clip` function ...
    boxes = RotatedBoxes(boxes.reshape(-1, B))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, B)  # R x C x B
    direct_scores = direct_scores.view(-1, num_bbox_reg_classes, 2) # R x C x 2
    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        direct_scores = direct_scores[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
        direct_scores = direct_scores[filter_mask]
    scores = scores[filter_mask]
    if direct_scores.shape[0] > 0:
        _, directions = direct_scores.max(dim=1)
    else:
        directions = torch.zeros_like(scores, dtype=torch.int64)

    # Apply per-class NMS
    # boxes_xyxy = BoxMode.convert(boxes[:,:-1], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    # keep = batched_nms(boxes_xyxy, scores, filter_inds[:, 1], nms_thresh)
    keep = batched_nms_rotated(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    directions = directions[keep]
    filter_inds[:, 0] = valid_inds[filter_inds[:, 0]]

    result = Instances(image_shape)
    result.pred_boxes = RotatedBoxes(boxes)
    result.scores = scores
    result.pred_directions = directions
    result.pred_classes = filter_inds[:, 1]

    return result, filter_inds[:, 0]

@ROI_HEADS_REGISTRY.register()
class MVDNetROIHeads(RROIHeads):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple([1.0 / input_shape[self.in_features[0]].stride])
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        assert pooler_type in ["ROIAlignRotated"]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_input_shape = {}
        for f in self.in_features:
            box_input_shape[f] = ShapeSpec(channels=input_shape[f].channels, height=pooler_resolution, width=pooler_resolution)

        self.box_head = build_box_head(
            cfg, box_input_shape
        )

        self.box_predictor = FastRCNNOutputLayers(
            input_size=self.box_head.output_size,
            num_classes=self.num_classes,
            cls_agnostic_bbox_reg=self.cls_agnostic_bbox_reg,
            box_dim=5,
        )
        
        self.direct_predictor = MVDNetRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        num_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            num_proposals.append(len(proposals_per_image))
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou_rotated(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[sampled_targets]
                proposals_per_image.gt_directions = targets_per_image.gt_directions[sampled_targets]
            else:
                gt_boxes = RotatedBoxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 5))
                )
                proposals_per_image.gt_boxes = gt_boxes
                gt_directions = torch.zeros_like(sampled_idxs)
                proposals_per_image.gt_directions = gt_directions

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        return super().forward(images, features, proposals, targets)

    def _forward_box(self, features, proposals):
        box_features = {}
        for f in self.in_features:
            if isinstance(features[f], torch.Tensor):
                box_features[f] = self.box_pooler([features[f]], [x.proposal_boxes for x in proposals])
            elif isinstance(features[f], list):
                box_features[f] = []
                for i in range(len(features[f])):
                    snapshot_features = [features[f][i]]
                    box_feature = self.box_pooler(snapshot_features, [x.proposal_boxes for x in proposals])
                    box_features[f].append(box_feature)
        
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        pred_direct_logits = self.direct_predictor(box_features)
        del box_features

        outputs = MVDNetRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            pred_direct_logits,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

class MVDNetRCNNOutputLayers(nn.Module):
    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg):
        super(MVDNetRCNNOutputLayers, self).__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        num_direct_score_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.direct_score = Linear(input_size, num_direct_score_classes * 2)
        nn.init.normal_(self.direct_score.weight, std=0.01)
        nn.init.constant_(self.direct_score.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.direct_score(x)
        return scores

class MVDNetRCNNOutputs(FastRCNNOutputs):
    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, pred_direct_logits, proposals, smooth_l1_beta
    ):
        super().__init__(box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta)
        self.pred_direct_logits = pred_direct_logits
        if len(proposals):
            if proposals[0].has("gt_boxes"):
                assert proposals[0].has("gt_directions")
                self.gt_directions = cat([p.gt_directions for p in proposals], dim=0)

    def softmax_cross_entropy_loss_direction(self):
        if self._no_instances:
            return 0.0 * F.cross_entropy(
                self.pred_direct_logits,
                torch.zeros(0, dtype=torch.long, device=self.pred_direct_logits.device),
                reduction="sum",
            )
        cls_agnostic_bbox_reg = self.pred_direct_logits.size(1) == 2
        device = self.pred_direct_logits.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(2, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = 2 * fg_gt_classes[:, None] + torch.arange(2, device=device)

        if len(fg_inds) == 0:
            return 0.0 * F.cross_entropy(
                self.pred_direct_logits,
                torch.zeros(self.pred_direct_logits.shape[0], dtype=torch.long, device=self.pred_direct_logits.device),
                reduction="sum",
            )
        loss_direct = F.cross_entropy(
            self.pred_direct_logits[fg_inds[:, None], gt_class_cols],
            self.gt_directions[fg_inds],
            reduction="sum"
        )
        loss_direct = loss_direct / self.gt_classes.numel()
        
        return loss_direct

    def losses(self):
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
            "loss_direct": self.softmax_cross_entropy_loss_direction(),
        }

    def predict_direction_probs(self):
        direction_probs = F.softmax(self.pred_direct_logits, dim=-1)
        return direction_probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        direct_scores = self.predict_direction_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference_direction(
            boxes, scores, direct_scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )