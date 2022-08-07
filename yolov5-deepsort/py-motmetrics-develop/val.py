import motmetrics as mm
import numpy as np
import os
#评价指标
metrics = list(mm.metrics.motchallenge_metrics)
#导入gt和ts文件
gt_file="./xxx/gt.txt"
ts_file="./xxx/ADL-Rundle-6.txt"
gt=mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
ts=mm.io.loadtxt(ts_file, fmt="mot15-2D")
name=os.path.splitext(os.path.basename(ts_file))[0]
#计算单个acc
acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=metrics, name=name)
print(mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names))
