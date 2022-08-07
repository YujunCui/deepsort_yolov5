import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracks_maintenance import AllTrackMaintenance

__all__ = ['DeepSort']  # __all__ 提供了暴露接口用的”白名单“


class DeepSort(object):
    def __init__(self,
                 model_path,            # 特征提取模型权重的路径
                 max_dist=0.2,          # 级联匹配的最大余弦距离阈值
                 min_confidence=0.3,    # 检测的置信度阈值
                 nms_max_overlap=1.0,   # 检测非极大抑制的阈值
                 max_iou_distance=0.7,  # 匈牙利算法的 IOU 阈值
                 max_age=70,            # 最大寿命
                 n_init=3,              # 最高击中次数
                 nn_budget=100,         # 最大保存特征帧数
                 use_cuda=True):        # 是否使用 gpu

        self.min_confidence = min_confidence    # 检测的置信度阈值
        self.nms_max_overlap = nms_max_overlap  # 检测非极大抑制的阈值，设置为1代表不进行抑制
        # 特征提取器(reid)，用于提取一个batch图片的对应特征
        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        # 距离度量方法，第一个参数可选 'cosine' or 'euclidean'
        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        # 初始化一个维护所有跟踪结果的对象
        self.tracks_maintain = AllTrackMaintenance(metric,
                                                   max_iou_distance=max_iou_distance,
                                                   max_age=max_age,
                                                   n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # 从原图中抠取 检测bbox 对应的图片，并通过特征提取模型计算得到相应的特征向量
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)  # 转换检测bbox格式
        # 筛选掉小于 min_confidence 的目标，并构造一个 Detection对象列表 detections
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)  # nms
        detections = [detections[i] for i in indices]

        # deepsort 跟踪的核心 2 步
        self.tracks_maintain.predict()           # 对所有跟踪轨迹执行卡尔曼预测过程
        self.tracks_maintain.update(detections)  # 执行测量更新和跟踪管理

        # 跟踪结果 output
        outputs = []
        for track in self.tracks_maintain.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        """
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
        """
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.

        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        """
        Convert bbox from xc_yc_w_h to x1_y1_x2_y2
        """
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)

        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        Convert bbox from xtl_ytl_w_h to x1_y1_x2_y2
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)

        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        """
        Convert bbox from x1_y1_x2_y2 to xtl_ytl_w_h
        """
        x1, y1, x2, y2 = bbox_xyxy
        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)

        return t, l, w, h

    # 获取抠图图像的特征
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]  # 获取抠图图像
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)  # 对抠图图像提取特征
        else:
            features = np.array([])

        return features
