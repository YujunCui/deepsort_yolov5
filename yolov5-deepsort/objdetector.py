# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:55:57 2022

@author: cuiyujun
"""

import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device


class Detector():
    def __init__(self):
        self.detector_path = 'weights/yolov5m.pt'          # 检测模型权重路径
        # self.obj_list = ['person']                         # 检测目标名称列表 (跟踪的目标名称, 根据需要添加)
        self.obj_list = ['person', 'car', 'bus', 'truck']  # 检测目标名称列表 (跟踪的目标名称, 根据需要添加)
        self.img_size = 640                                # 检测输入图像的尺寸
        self.threshold = 0.3                               # 检测器置信度阈值
        self.iou_thres = 0.4                               # 检测器nms的iou阈值
        self.device = select_device('0' if torch.cuda.is_available() else 'cpu')  # cpu or gpu
        self.m = self.init_model()                         # 初始化模型
        self.names = self.m.module.names if hasattr(self.m, 'module') else self.m.names  # 检测模型的 所有目标名称列表

    # 初始化模型
    def init_model(self):
        model = attempt_load(self.detector_path, map_location=self.device).to(self.device) # 载入模型
        model.eval()  # 模型进入推理模式，主要是固定 BN 和 Dropout.
        model.half()  # 模型半精度化，提高推理速度

        return model

    # 预处理
    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]  # 等比例缩放
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)                   # 将一个内存不连续存储的数组转换为内存连续存储的数组，加快运行速度
        img = torch.from_numpy(img).to(self.device)
        img = img.half()                                  # 输入图像半精度
        img /= 255.0                                      # 输入图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)                        # 扩展数据维度

        return img0, img                                  # 返回原图和送入检测器的图像

    # 检测，包括前后处理
    def detect(self, im):
        im0, img = self.preprocess(im)        # 预处理
        pred = self.m(img, augment=False)[0]  # 检测器前向过程
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, self.iou_thres)  # 非极大值抑制过程
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in self.obj_list:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes  # 返回原图和原图对应自定义格式的检测框


if __name__ == '__main__':
    detector = Detector()  # 初始化检测器
    import cv2
    image = cv2.imread(r'D:\AI\Completed_Projects\Track\deepsort\yolov5-deepsort\images\zidane.jpg')
    im, bboxes = detector.detect(image)
    for i in range(len(bboxes)):
        cv2.rectangle(im, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), color=(0, 0, 255), thickness=2)
    cv2.imshow('img', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
