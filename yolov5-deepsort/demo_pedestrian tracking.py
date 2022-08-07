# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:55:57 2022

@author: cuiyujun
"""

import os
import cv2
from pathlib import Path
from objdetector import Detector
from objtracker import MOTracker


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
video_format = ['.mp4', '.ts']


def main(trackvideodir, save_result=False, save_txt=False):
    if save_result:
        savedir = './logs'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    trackvideos_path = []
    for format in video_format:
        trackvideos_path += list(Path(trackvideodir).rglob('*' + format))

    for VIDEO_PATH in trackvideos_path:
        if save_result:
            RESULT_PATH = os.path.join(savedir, VIDEO_PATH.stem + '_result.mp4')
        if save_txt:
            track_result_saver = open(os.path.join(savedir, VIDEO_PATH.stem + '_result.txt'), 'w')

        detector = Detector()  # 初始化检测器
        tracker = MOTracker()  # 初始化跟踪器

        cap = cv2.VideoCapture(str(VIDEO_PATH))  # 根据视频路径, 读取视频
        fps = int(cap.get(5))  # 获取帧率
        print('FPS: ', fps)

        video_writer = None  # 写视频
        name = 'demo'
        skip_num = 5  # 跳帧数
        n = 0  # 帧数计数器
        while True:
            if n % skip_num == 0:
                _, im = cap.read()  # 读帧
                if im is None:
                    break

                result, bboxes = tracker.track(detector, im)
                h, w = result.shape[:2]
                result = cv2.resize(result, (int(500*w/h), 500))  # 缩放画面

                for box in bboxes:
                    track_id, x1, x2, w, h = box[-1], box[0], box[1], box[2] - box[0], box[3] - box[1]
                    track_result_saver.write(f'{n+1},{track_id},{x1},{x2},{w},{h},1，-1，-1，-1\n')
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 视频编码器 opencv3.0
                    video_writer = cv2.VideoWriter(RESULT_PATH,  # 保存视频的路径
                                                   fourcc,       # 指定视频编码器
                                                   fps,          # 保存帧率
                                                   (result.shape[1], result.shape[0]))  # 保存视频宽、高

                video_writer.write(result)  # 写帧
                cv2.imshow(name, result)
                cv2.waitKey(int(1000/fps))

                if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                    break
            n += 1
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        track_result_saver.close()


if __name__ == '__main__':
    trackvideodir = './video/mot'
    main(trackvideodir, save_result=True, save_txt=True)
