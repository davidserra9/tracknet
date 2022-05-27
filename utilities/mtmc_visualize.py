import os
import time
from os.path import join

import cv2
import numpy as np
from matplotlib import pyplot as plt

from dataset_utils import load_annot
from image_utils import plotBBoxes

DATA_ROOT = '../../../data/AICity_data/train'
PRED_ROOT = '../data/fasterrcnn/S01-S04/mtmc_max_overlap'
SEQ = 'S03'

# Offset frames
timestamps = {'c010': 0,
              'c011': 3,
              'c012': 28,
              'c013': 87,
              'c014': 37,
              'c015': 2,
              }

seq_annot = {}
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions

for cam in sorted(os.listdir(PRED_ROOT)):
    seq_annot[cam.split('.txt')[0]] = load_annot(join(PRED_ROOT), cam)

num_frames = []
for cam in sorted(os.listdir(join(DATA_ROOT, SEQ))):
    num_frames.append(len(os.listdir(join(DATA_ROOT, SEQ, cam, 'frames'))))

min_num_frames = min(num_frames)

for frame_num in range(min_num_frames):
    frame_imgs = []
    for cam_name, cam_annot in seq_annot.items():
        img = cv2.imread(join(DATA_ROOT, SEQ, cam_name, 'frames', f'{(frame_num+timestamps[cam_name]):04}' + '.jpg'))
        frame_annotations = cam_annot.get(f'{(frame_num+timestamps[cam_name]):04}', [])
        frame_bboxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][2], a['bbox'][3], a['obj_id']] for a in frame_annotations]
        img = plotBBoxes(img, frame=(frame_num+timestamps[cam_name]), cam=cam_name,saveFrames=None, mtmc=frame_bboxes)
        img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(img, (0, 0), (640, 480), (0, 0, 0), thickness=2)
        frame_imgs.append(img)

    joined_img = np.hstack((
        np.vstack((frame_imgs[4], frame_imgs[3])),
        np.vstack((frame_imgs[5], frame_imgs[2])),
        np.vstack((frame_imgs[0], frame_imgs[1]))
    ))
    cv2.imshow('frame', joined_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
