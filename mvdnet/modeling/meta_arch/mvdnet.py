import numpy as np
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.modeling import(
    META_ARCH_REGISTRY, 
    GeneralizedRCNN, 
    detector_postprocess,
    build_backbone,
    build_proposal_generator,
    build_roi_heads
)

@META_ARCH_REGISTRY.register()
class MVDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.history_on = cfg.INPUT.HISTORY_ON
        
        self.to(self.device)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        radar_data, lidar_data = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(radar_data.tensor, lidar_data.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(radar_data, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(radar_data, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        radar_data, lidar_data = self.preprocess_image(batched_inputs)
        features = self.backbone(radar_data.tensor, lidar_data.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(radar_data, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(radar_data, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            if detected_instances is None:
                return MVDNet._postprocess(results, proposals, batched_inputs, radar_data.image_sizes)
            else:
                return GeneralizedRCNN._postprocess(results, batched_inputs, radar_data.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        if self.history_on:
            radar_data = [torch.stack(x["radar_intensity"]).to(self.device).type(torch.float) for x in batched_inputs]
            lidar_data = [torch.cat([torch.stack(x["lidar_intensity"]), torch.stack(x["lidar_occupancy"])], dim=1).to(self.device).type(torch.float) for x in batched_inputs]
        else:
            radar_data = [x["radar_intensity"].to(self.device).type(torch.float) for x in batched_inputs]
            lidar_data = [torch.cat([x["lidar_intensity"], x["lidar_occupancy"]], dim=0).to(self.device).type(torch.float) for x in batched_inputs]
        radar_data = ImageList.from_tensors(radar_data, self.backbone.size_divisibility)
        lidar_data = ImageList.from_tensors(lidar_data, self.backbone.size_divisibility)
        return radar_data, lidar_data

    @staticmethod
    def _postprocess(instances, proposals, batched_inputs, image_sizes):
        processed_results = []
        for results_per_image, proposals_per_image, input_per_image, image_size in zip(
            instances, proposals, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            p = detector_postprocess(proposals_per_image, height, width)
            processed_results.append({"instances": r, "proposals": p})
        return processed_results