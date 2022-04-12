# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from .predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

class ObjectDetection(object):
    def __init__(
        self,
        config_file = None,
        model_file = None,
        opts = [],
        confidence_threshold = 0.8
    ):
        if not config_file:
            path = os.path.dirname(__file__)
            config_file = path + "/../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        if not model_file:
            path = os.path.dirname(__file__)
            model_file = path + "/models/mask_rcnn_R_50_FPN_3x.pkl"
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_file
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.freeze()
        mp.set_start_method("spawn", force=True)
        self.demo = VisualizationDemo(cfg)
        self.logger = setup_logger()

    def inference(self, img):
        predictions, visualized_output = self.demo.run_on_image(img)
        return predictions, visualized_output
    
def image(object_detection, image, output):
    img = cv2.imread(image)
    start_time = time.time()
    predictions, visualized_output = object_detection.inference(img)
    object_detection.logger.info(
        "{} in {:.2f}s".format(
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )
    visualized_output.save(output)
    

def main(args):
    object_detection = ObjectDetection()
    image(object_detection, "./image/dog.jpg", "./result.jpg") 

if __name__ == "__main__":
    main(args)