import os
import os.path as osp

DATASET_DIR = '/home/adam/workspace/venv/bypy/synthetic_chinese_string'

all_labels = []
with open(osp.join(DATASET_DIR, 'train.txt')) as f:
    lines = f.read().split('\n')
    for line in lines:
        if line:
            labels = [int(label) for label in line.split(' ')[1:]]
            all_labels.append(labels)
import numpy as np
print(np.max(np.array(all_labels)))

