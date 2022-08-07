# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:55:57 2022

@author: cuiyujun
"""

import cv2
import torch
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class MOTracker(object):
    def __init__(self):
        self.tracker_path = "deep_sort/configs/deep_sort.yaml"  # 跟踪配置文件的路径
        self.cfg = get_config()                                 # 初始化跟踪配置文件读取方式
        self.cfg.merge_from_file(self.tracker_path)             # 读取跟踪配置文件
        # 跟踪算法 deepsort
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,                          # 特征提取模型权重的路径
                                 max_dist=self.cfg.DEEPSORT.MAX_DIST,                  # 级联匹配的最大余弦距离阈值
                                 min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,      # 检测的置信度阈值
                                 nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,    # 检测非极大抑制的阈值
                                 max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,  # 匈牙利算法的 IOU 阈值
                                 max_age=self.cfg.DEEPSORT.MAX_AGE,                    # 最大寿命
                                 n_init=self.cfg.DEEPSORT.N_INIT,                      # 最高击中次数
                                 nn_budget=self.cfg.DEEPSORT.NN_BUDGET,                # 最大保存特征帧数
                                 use_cuda=True)                                        # 是否使用 gpu

    # 绘制跟踪图像
    def plot_bboxes(self, image, bboxes, line_thickness=None, line_width=3, txt_color=(255, 255, 255)):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        list_pts = []
        point_radius = 4

        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            if cls_id in ['smoke', 'phone', 'eat']:
                color = (0, 0, 255)
            else:
                color = (59, 159, 238)
            if cls_id == 'eat':
                cls_id = 'eat-drink'

            # check whether hit line
            check_point_x = x1
            check_point_y = int(y1 + ((y2 - y1) * 0.6))

            p1, p2 = (x1, y1), (x2, y2)
            lw = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
            cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            text = '{} ID{}'.format(cls_id, pos_id)
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(text, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                        text, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

            list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
            list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
            list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
            list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

            ndarray_pts = np.array(list_pts, np.int32)
            cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
            list_pts.clear()

        return image

    # 跟踪过程
    def track(self, target_detector, image):
            _, bboxes = target_detector.detect(image)  # 利用检测器获取检测框结果
            bbox_xywh = []
            confs = []
            bboxes2draw = []
            if len(bboxes):
                # Adapt detections to deep sort input format
                for x1, y1, x2, y2, _, conf in bboxes:
                    obj = [int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1]
                    bbox_xywh.append(obj)
                    confs.append(conf)
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # 对检测结果进行跟踪，并返回跟踪结果 outputs
                outputs = self.deepsort.update(xywhs, confss, image)
                for value in list(outputs):
                    x1,y1,x2,y2,track_id = value
                    bboxes2draw.append((x1, y1, x2, y2, '', track_id))
            image = self.plot_bboxes(image, bboxes2draw)  # 绘制跟踪图像

            return image, bboxes2draw  # 返回绘制的跟踪图像 和 跟踪结果
