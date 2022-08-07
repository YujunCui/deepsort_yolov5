# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:55:57 2022

@author: cuiyujun
"""

import os
import cv2
import time
from tqdm import tqdm
from pathlib import Path


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
image_format = ['.jpg', '.jepg', '.png']


def main(imgsdir):
    savedir = './logs'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    imgs_path = []
    for format in image_format:
        imgs_path += list(Path(imgsdir).rglob('*' + format))
    imgs_path.sort()

    RESULT_PATH = os.path.join(savedir, os.path.basename(imgsdir) + '.mp4')

    fps = 25  # 获取帧率
    print('FPS: ', fps)
    video_writer = None  # 写视频
    pbar = tqdm(imgs_path)
    for img_path in pbar:
        img = cv2.imread(str(img_path))
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 视频编码器 opencv3.0
            video_writer = cv2.VideoWriter(RESULT_PATH,  # 保存视频的路径
                                           fourcc,       # 指定视频编码器
                                           fps,          # 保存帧率
                                           (img.shape[1], img.shape[0]))  # 保存视频宽、高

        video_writer.write(img)  # 写帧
        time.sleep(0.01)

    video_writer.release()
    print(f'Video is saved at {RESULT_PATH}!')


if __name__ == '__main__':
    imgsdir = r'D:\Data\track\MOT16_source_compressed_MOT16\train\MOT16-05\img1'
    main(imgsdir)
