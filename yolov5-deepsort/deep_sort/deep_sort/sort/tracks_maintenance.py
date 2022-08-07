# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class AllTrackMaintenance():
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
        测量与轨迹关联的距离度量
    max_age : int
        Maximum number of missed misses before a track is deleted.
        删除轨迹前的最大未命中数
    n_init : int
        Number of frames that a track remains in initialization phase.
        确认轨迹前的连续检测次数。如果前n_init帧内发生未命中，则将轨迹状态设置为Deleted
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric                      # 距离度量方法
        self.max_iou_distance = max_iou_distance  # 匈牙利算法的 IOU 阈值
        self.max_age = max_age                    # 最大寿命
        self.n_init = n_init                      # 最高击中次数
        self.kf = kalman_filter.KalmanFilter()    # 实例化卡尔曼滤波器
        self.tracks = []                          # 轨迹维护列表，用于保存一系列轨迹
        self._next_id = 1                         # 下一个新增轨迹的 id 编号

    def predict(self):
        """Propagate track state distributions one time step forward.
        将跟踪状态分布向前传播一步

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)  # 对单个跟踪轨迹执行卡尔曼的预测过程(包括状态和协方差预测)等

    def update(self, detections):
        """Perform measurement update and track management.
        执行测量更新和轨迹管理

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # 进行匹配 (包括 级联匹配 和 匈牙利匹配)
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # 更新跟踪结果
        # 1. 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # 对相应的跟踪轨迹 执行卡尔曼的更新过程; 将跟踪轨迹的特征向量添加到跟踪轨迹的特征的集合中用于后续余弦距离计算; 以及更新跟踪轨迹状态
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        # 2. 针对 track 失配
        for track_idx in unmatched_tracks:
            # 更新删除态，若 Tantative 则更新为删除态；若 update 时间很久也更新为删除态
            self.tracks[track_idx].mark_missed()
        # 3. 针对 detection 失配
        for detection_idx in unmatched_detections:
            # 初始化新的轨迹
            self._initiate_track(detections[detection_idx])
        
        # 剔除删除态的轨迹 (即保存的是标记为 Confirmed 和 Tentative 的轨迹)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 更新距离度量
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]  # Confirmed 状态 track 的 id 列表
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # 将 Confirmed 状态 track 的 features 添加到 features 列表
            targets += [track.track_id for _ in track.features]  # 获取每个 feature 对应 track 的 id
            track.features = []  # 清空跟踪轨迹的特征向量列表
        # 距离度量中特征向量集的更新
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # 级联匹配的代价矩阵
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])  # 检测特征
            targets = np.array([tracks[i].track_id for i in track_indices])    # 跟踪 id
            
            # 使用余弦距离计算代价矩阵 (外观特征)
            cost_matrix = self.metric.distance(features, targets)
            # 如果要一个轨迹去匹配两个外观特征相似的 detection，也很容易出错；
            # 使用马氏距离 (运动特征) 过滤外观特征相似但距离较远的 detection，从而减少错误的匹配。
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, dets,
                                                             track_indices, detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 将轨迹分为确定态和不确定态
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # 对确定态的轨迹进行级联匹配 (外观和运动特征匹配)，得到匹配的tracks、不匹配的tracks、不匹配的detections
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                                                              gated_metric, self.metric.matching_threshold, self.max_age,
                                                              self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.        
        # 将 不确定态的轨迹 和 级联匹配中刚刚没有匹配上的轨迹 组合为 iou匹配候选集合 iou_track_candidates
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        # 级联匹配中并非刚刚没有匹配上的轨迹
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        # 对级联匹配中还没有匹配成功的目标, 使用匈牙利算法进行IoU匹配
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                                                              iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                                                              detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b  # 组合两部分匹配结果
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))  # 组合两部分未匹配结果

        return matches, unmatched_tracks, unmatched_detections

    # 初始化新轨迹
    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())  # 初始化新轨迹的卡尔曼滤波初始状态和协方差
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))  # 添加新轨迹
        self._next_id += 1  # 下一个新增 id 编号，加 1
