# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
This evaluator is meant for regular COCO mAP evaluation, for example on the COCO val set.

For Category mAP, we need the model to make predictions for all the categories on every single image.
In general, since the number of classes can be big, and the API model makes predictions individually for each pair (image, class),
we may need to split the inference process for a given image in several chunks.
"""

import logging
from collections import defaultdict
from typing import Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.train.utils.distributed import is_main_process

try:
    from tidecv import datasets, TIDE

    HAS_TIDE = True
except ImportError:
    HAS_TIDE = False
    logging.debug("TIDE not installed. Install via `pip install tidecv` for detailed error analysis.")


# the COCO detection metrics (https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L460-L471)
COCO_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def convert_to_xywh(boxes):
    """Convert bounding boxes from xyxy format to xywh format."""
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


def _resize_masks(anns, target_resolution):
    """Resize RLE-encoded masks to a fixed square resolution."""
    import pycocotools.mask as mask_utils
    import numpy as np

    target_h = target_w = target_resolution
    for ann in anns:
        rle = ann["segmentation"]
        h, w = rle["size"]
        if h == target_h and w == target_w:
            continue
        mask = mask_utils.decode(rle)
        row_idx = np.linspace(0, h - 1, target_h, dtype=int)
        col_idx = np.linspace(0, w - 1, target_w, dtype=int)
        mask_resized = mask[np.ix_(row_idx, col_idx)]
        ann["segmentation"] = mask_utils.encode(np.asfortranarray(mask_resized))
        ann["area"] = int(mask_resized.sum())


def _build_downscaled_gt_json(gt_path: str, eval_resolution: int, out_path: str) -> str:
    """Read a COCO GT JSON, downscale all annotation masks to (eval_resolution x
    eval_resolution), and write a new JSON to out_path. Handles both polygon
    and RLE segmentations. Needed for TIDE because it loads the GT file
    directly and has no hook for in-memory mask resizing (unlike pycocotools
    where _prepare() intercepts the annotations first).
    """
    import json
    import numpy as np
    import pycocotools.mask as mask_utils

    target_h = target_w = eval_resolution
    with open(gt_path, "r") as f:
        data = json.load(f)

    id_to_img = {img["id"]: img for img in data.get("images", [])}

    for ann in data.get("annotations", []):
        img = id_to_img.get(ann["image_id"])
        if img is None:
            continue
        src_h, src_w = img["height"], img["width"]
        seg = ann.get("segmentation")
        if seg is None:
            continue
        # Decode any supported format to a binary mask at source resolution.
        if isinstance(seg, list):
            # polygon list
            rles = mask_utils.frPyObjects(seg, src_h, src_w)
            rle = mask_utils.merge(rles) if len(rles) > 1 else rles[0]
            mask = mask_utils.decode(rle)
        elif isinstance(seg, dict):
            if isinstance(seg.get("counts"), list):
                # uncompressed RLE
                rle = mask_utils.frPyObjects(seg, src_h, src_w)
            else:
                rle = seg
            mask = mask_utils.decode(rle)
        else:
            continue
        # Nearest-neighbor downscale via index mapping.
        row_idx = np.linspace(0, src_h - 1, target_h, dtype=int)
        col_idx = np.linspace(0, src_w - 1, target_w, dtype=int)
        mask_resized = mask[np.ix_(row_idx, col_idx)]
        rle_resized = mask_utils.encode(np.asfortranarray(mask_resized))
        # `counts` is bytes after encode — make JSON-serializable.
        if isinstance(rle_resized.get("counts"), (bytes, bytearray)):
            rle_resized["counts"] = rle_resized["counts"].decode("ascii")
        ann["segmentation"] = rle_resized
        ann["area"] = int(mask_resized.sum())
        # Box coords also need to be scaled so TIDE's area categories stay sane.
        if "bbox" in ann and ann["bbox"] is not None:
            x, y, w, h = ann["bbox"]
            sx = target_w / src_w
            sy = target_h / src_h
            ann["bbox"] = [x * sx, y * sy, w * sx, h * sy]

    # Images must also report the new size so downstream tools pick it up.
    for img in data.get("images", []):
        img["height"] = target_h
        img["width"] = target_w

    with open(out_path, "w") as f:
        json.dump(data, f)
    return out_path


class HeapElement:
    """Utility class to make a heap with a custom comparator"""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class COCOevalCustom(COCOeval):
    """
    This is a slightly modified version of the original COCO API with added support for positive split evaluation.
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="segm", dt_only_positive=False,
        eval_resolution=None,
    ):
        super().__init__(cocoGt, cocoDt, iouType)
        self.dt_only_positive = dt_only_positive
        # eval_resolution is kept for backwards-compat but is no longer the
        # source of truth. _prepare() now auto-detects the prediction
        # resolution from the dt RLEs and resizes GT to match, so the same
        # evaluator works for both low-res training eval and full-res final
        # eval without config changes.
        self.eval_resolution = eval_resolution

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
            # Auto-detect prediction resolution from the first dt RLE and
            # downscale GT masks to match. Handles both low-res training eval
            # (pred res == eval_resolution, e.g. 1008) and full-res final eval
            # (pred res == original image size -> GT already matches, no-op).
            if dts and gts:
                seg = dts[0].get("segmentation")
                if isinstance(seg, dict) and "size" in seg:
                    pred_h, pred_w = seg["size"]
                    if pred_h == pred_w:
                        gt_seg = gts[0].get("segmentation")
                        gt_size = gt_seg.get("size") if isinstance(gt_seg, dict) else None
                        if gt_size is None or tuple(gt_size) != (pred_h, pred_w):
                            _resize_masks(gts, pred_h)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        _gts_cat_ids = defaultdict(set)  # gt for evaluation on positive split
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
            _gts_cat_ids[gt["image_id"]].add(gt["category_id"])

        #### BEGIN MODIFICATION ####
        for dt in dts:
            if (
                self.dt_only_positive
                and dt["category_id"] not in _gts_cat_ids[dt["image_id"]]
            ):
                continue
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        #### END MODIFICATION ####
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results


class CocoEvaluatorOfflineWithPredFileEvaluators:
    def __init__(
        self,
        gt_path,
        tide: bool = True,
        iou_type: str = "bbox",
        positive_split=False,
        eval_resolution: Optional[int] = None,
    ):
        self.gt_path = gt_path
        self.tide_enabled = HAS_TIDE and tide
        self.positive_split = positive_split
        self.iou_type = iou_type
        self.eval_resolution = eval_resolution
        # Cache for the downscaled GT built for TIDE. TIDE loads the GT file
        # directly (no hook for in-memory resizing), so when predictions come
        # back at reduced resolution we must feed it a pre-resized file or IoUs
        # will all be 0. _downscaled_gt_path holds the cached resolution (int),
        # _downscaled_gt_file the on-disk path. Rebuild only if the prediction
        # resolution changes between calls (e.g. training's 1008 -> final
        # eval's full res).
        self._downscaled_gt_path: Optional[int] = None
        self._downscaled_gt_file: Optional[str] = None

    def _detect_pred_resolution(self, dumped_file) -> Optional[int]:
        """Inspect the first prediction in dumped_file and return its square
        mask resolution (h when h == w). Returns None if the file is empty or
        malformed. Used to decide whether GT needs to be downscaled for TIDE
        (low-res train-time eval) or used as-is (full-res final eval)."""
        import json

        try:
            with open(str(dumped_file), "r") as f:
                preds = json.load(f)
        except Exception:
            return None
        if not preds:
            return None
        seg = preds[0].get("segmentation")
        if not isinstance(seg, dict):
            return None
        size = seg.get("size")
        if not size or size[0] != size[1]:
            return None
        return int(size[0])

    def _get_tide_gt_path(self, dumped_file) -> str:
        """Return the GT path TIDE should load. Inspects the prediction file
        to determine the resolution; if predictions are at a reduced resolution
        (typical for low-res training eval), builds (once) a downscaled GT
        JSON at that resolution. Otherwise returns the original GT path."""
        pred_res = self._detect_pred_resolution(dumped_file)
        if pred_res is None or pred_res <= 0:
            return self.gt_path
        # If predictions match original GT resolution, no downscaling needed.
        # We don't know original dims without loading GT, so just skip when
        # pred_res is "large" (>2048) — full-res drone images are 3648+.
        if pred_res > 2048:
            return self.gt_path
        if self._downscaled_gt_path != pred_res:
            import os

            base, ext = os.path.splitext(self.gt_path)
            out_path = f"{base}.downscaled_{pred_res}{ext}"
            if not os.path.exists(out_path):
                logging.info(
                    f"TIDE: building downscaled GT at {pred_res}x{pred_res} -> {out_path}"
                )
                _build_downscaled_gt_json(self.gt_path, pred_res, out_path)
            self._downscaled_gt_path = pred_res
            self._downscaled_gt_file = out_path
        return self._downscaled_gt_file

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("OfflineCoco evaluator: Loading groundtruth")
        self.gt = COCO(self.gt_path)

        # Creating the result file
        logging.info("Coco evaluator: Creating the result file")
        cocoDt = self.gt.loadRes(str(dumped_file))

        # Run the evaluation
        logging.info("Coco evaluator: Running evaluation")
        coco_eval = COCOevalCustom(
            self.gt, cocoDt, iouType=self.iou_type, dt_only_positive=self.positive_split,
            eval_resolution=self.eval_resolution,
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        outs = {}
        for i, value in enumerate(coco_eval.stats):
            outs[f"coco_eval_{self.iou_type}_{COCO_METRICS[i]}"] = value

        if self.tide_enabled:
            logging.info("Coco evaluator: Loading TIDE")
            tide_gt_path = self._get_tide_gt_path(dumped_file)
            self.tide_gt = datasets.COCO(tide_gt_path)
            self.tide = TIDE(mode="mask" if self.iou_type == "segm" else "bbox")

            # Run TIDE
            logging.info("Coco evaluator: Running TIDE")
            self.tide.evaluate(
                self.tide_gt, datasets.COCOResult(str(dumped_file)), name="coco_eval"
            )
            self.tide.summarize()
            for k, v in self.tide.get_main_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

            for k, v in self.tide.get_special_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

        return outs
