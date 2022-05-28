import cv2
import time

from scipy import ndimage

from dataset_utils import load_annot
from image_utils import plotBBoxes
from os.path import join

from image_utils import filter_roi

DATA_ROOT = '../../../data/AICity_data/train'
SEQ = 'S03'
CAM = 'c011'

#labels = load_annot(join(DATA_ROOT, SEQ, CAM, 'gt'), 'gt.txt')

labels = load_annot(join('..', 'data', 'fasterrcnn', 'S01-S04', 'mtsc_deep_sort'), 'c011.txt')
video = cv2.VideoCapture(join(DATA_ROOT, SEQ, CAM, 'vdo.avi'))

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
roi = cv2.imread(join(DATA_ROOT, SEQ, CAM, "roi.jpg"), cv2.IMREAD_GRAYSCALE) / 255
roi = ndimage.distance_transform_edt(roi)
count = 1
while True:
    ret, frame = video.read()

    if ret:
        # Draw the frame number on the frame
        cv2.putText(frame, str(count), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Obtain the frame BBoxes
        annotations = labels.get(f'{count:04}',[])   # To avoid crashing when the frame does not contain annotations

        if len(annotations) > 0:
            bboxes = [annotation['bbox'] for annotation in annotations]
            obj_ids = [annotation['obj_id'] for annotation in annotations]
            detections = [[box[0], box[1], box[2], box[3], obj_id] for box, obj_id in zip(bboxes, obj_ids)]
            detections = filter_roi(detections, roi, th=100)
            plotBBoxes(frame, saveFrames=None, annotations=detections)

        cv2.imshow('frame', frame)
        time.sleep(0.05)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break