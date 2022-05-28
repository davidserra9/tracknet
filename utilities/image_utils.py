import os
import cv2
import motmetrics as mm
import sys

import numpy as np
from tqdm import tqdm
from os.path import dirname, join, exists
from sort.sort import Sort
from ByteTrack.yolox.deepsort_tracker.deepsort import DeepSort
from ByteTrack.yolox.deepsort_tracker.deepsort import DeepSort
def video_to_frames(video_path):
    """
    Read and save a single video from the given path. The frames are saved in the same camera folder in a new folder
    named frames. i.e. is the input is the /S03/c010/vdo.avi this function will save all the in a new folder
    /S03/c010/frames/ which will contain:
    /S03/c010/frames/
            |---0000.jpg
            |---0001.jpg
            |   ...
            |---2174.jpg
            |---2175.jpg
    :param
        video_path: path to the video (../../data/AICity_data/train/S03/c010/vdo.avi)
    """

    # Create the frames' folder inside the camera folder
    camera_path = dirname(video_path)

    # If the folder frames do not exists, it means that the frames have not been extracted
    if not exists(join(camera_path, "frames")):
        os.makedirs(join(camera_path, "frames"), exist_ok=True)

        # Open video capture
        cap = cv2.VideoCapture(video_path)

        # Check if video is opened
        if not cap.isOpened():
            raise IOError("Could not open video")

        # Create the tqdm progress bar object
        frame_num = 0
        pbar = tqdm(desc=f"Reading and saving frames from {camera_path}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save frame
            cv2.imwrite(join(camera_path, "frames", f"{frame_num:04d}.jpg"), frame)
            pbar.update(1)
            frame_num += 1

        # Release video
        cap.release()

    else:
        print(f"Frames of {camera_path.split('/')[-2]}/{camera_path.split('/')[-1]} already saved...")


def all_videos_to_frames(data_root="../../data/AICity_data/train"):
    """
    Read and save all the videos from the dataset in individual frames using the video_to_frames function.
    :param
    data_root: path to the dataset (../../data/AICity_data/train)
    """
    # Get a list of all the videos in the dataset
    video_paths = [join(data_root, seq, cam, 'vdo.avi') for seq in os.listdir(data_root) for cam in os.listdir(join(data_root, seq))]
    video_paths = sorted(video_paths)

    # For each video, read and save the frames
    for video in video_paths:
        video_to_frames(video)


def compute_bbox_centroid(bbox):
    """
    Compute the centroid of a bounding box.
    :param
        bbox: bounding box (xmin, ymin, xmax, ymax)
    :return
        centroid: (x, y)
    """
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


def filter_roi(bboxes, roi_dist, th=50):
    """
    Filter the bounding boxes that are too close to the ROI.
    :param bboxes: list of bounding boxes (xmin, ymin, xmax, ymax)
    :param roi_dist: ndimage.distance object in which each pixel has the distance to the ROI
    :param th: int, distance in which the bboxes are discarted
    :return: list, filtered bounding boxes
    """
    filtered_bboxes = []
    for bbox in bboxes:
        centroid = compute_bbox_centroid(bbox)
        if roi_dist[centroid[1], centroid[0]] > th:
            filtered_bboxes.append(bbox)
    return filtered_bboxes


def iou_list(bboxes1, bbox2):
    """
    Computes IoU between a list of bounding boxes and a single bounding box.
    :param bboxes1: list of bounding boxes [xmin, ymin, xmax, ymax]
    :param bbox2: 1 bounding box [xmin, ymin, xmax, ymax]
    :return: list of IoU (size num)
    """

    # intersection
    xA = np.maximum(bboxes1[:, 0], bbox2[0])
    yA = np.maximum(bboxes1[:, 1], bbox2[1])
    xB = np.minimum(bboxes1[:, 2], bbox2[2])
    yB = np.minimum(bboxes1[:, 3], bbox2[3])
    iw = np.maximum(xB - xA + 1., 0.)
    ih = np.maximum(yB - yA + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bbox2[2] - bbox2[0] + 1.) * (bbox2[3] - bbox2[1] + 1.) +
           (bboxes1[:, 2] - bboxes1[:, 0] + 1.) *
           (bboxes1[:, 3] - bboxes1[:, 1] + 1.) - inters)

    return inters / uni


def tracking(img_paths, ground_truth, predictions, type='sort', roi=None, roi_th=50, starting_id=1):
    """
    Compute the tracking and its evaluation given a list of predictions and ground truths.
    This function is able to do the tracking with and without filtering the detections by a mask.
    If a parameter roi is passed, the tracking will be done only in the region of interest.
    :param img_paths: list of image paths corresponding the predictions
    :param ground_truth: list of ground truth bounding boxes in the format of load_annot
    :param predictions: list of predictions bounding boxes in the format of load_annot
    :param roi: ndimage.distance object
    :param roi_th: mask threshold
    :return: list of predictions with the corresponding obj_id for each detection
    :return: idf1 score of the corresponding sequence
    """
    if type == 'sort':
        # Create the accumulator that will be updated during each frame
        accumulator = mm.MOTAccumulator(auto_id=True)

        starting_id = None
        # Create the tracker
        tracker = Sort()

        tracking_predictions = []
        # Iterate through the frames
        for img_path in tqdm(img_paths, desc=f"Tracking {img_paths[0].split('/')[-3]}"):

            frame_num = img_path.split('/')[-1].split('.')[0]

            # Obtain the Ground Truth and predictions for the current frame
            # Using the function get() to avoid crashing when there is no key with that string
            gt_annotations = ground_truth.get(frame_num, [])
            pred_annotations = predictions.get(frame_num, [])

            # Obtain the Ground Truth and predictions for the current frame
            gt_bboxes = [anno['bbox'] for anno in gt_annotations]
            pred_bboxes = [anno['bbox'] for anno in pred_annotations]
            pred_scores = [anno['confidence'] for anno in pred_annotations]
            pred_bboxes = [[box[0], box[1], box[2], box[3], score] for box, score in
                           zip(pred_bboxes, pred_scores)]  # Convert to list

            # If roi is not None, filter predictions
            if roi is not None:
                pred_bboxes = filter_roi(bboxes=pred_bboxes, roi_dist=roi, th=roi_th)

            # Obtain the Ground Truth centers and track IDs
            gt_centers = [(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in gt_bboxes]
            gt_ids = [anno['obj_id'] for anno in gt_annotations]

            # Update tracker
            if len(pred_bboxes) == 0:
                trackers = tracker.update(np.empty((0, 5)))
            else:
                trackers = tracker.update(np.array(pred_bboxes))

            det_centers = []
            det_ids = []

            for t in trackers:
                det_centers.append((int(t[0] + t[2] / 2), int(t[1] + t[3] / 2)))
                det_ids.append(int(t[4]))
                tracking_predictions.append(
                    [frame_num, int(t[4]), int(t[0]), int(t[1]), int(t[2] - t[0]), int(t[3] - t[1]), 1])

            accumulator.update(
                gt_ids,  # Ground truth objects in this frame
                det_ids,  # Detector hypotheses in this frame
                mm.distances.norm2squared_matrix(gt_centers, det_centers)
                # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
            )

        # Compute the metrics
        mh = mm.metrics.create()
        summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
        print(summary)

    elif type == 'max_overlap':
        # Create the accumulator that will be updated during each frame
        accumulator = mm.MOTAccumulator(auto_id=True)

        tracking_predictions = []
        initialize = True

        for img_path in tqdm(img_paths, desc=f"Tracking {img_paths[0].split('/')[-3]}"):
            frame_num = img_path.split('/')[-1].split('.')[0]

            # Obtain the Ground Truth and predictions for the current frame
            # Using the function get() to avoid crashing when there is no key with that string
            gt_annotations = ground_truth.get(frame_num, [])
            pred_annotations = predictions.get(frame_num, [])

            # Obtain the Ground Truth centers and ids
            gt_centers = [(a['bbox'][0] + a['bbox'][2] / 2, a['bbox'][1] + a['bbox'][3] / 2) for a in gt_annotations]
            gt_ids = [a['obj_id'] for a in gt_annotations]

            pred_centers = []
            pred_ids = []
            current_bboxes = []

            frame_pred_bboxes = [a['bbox'] for a in pred_annotations]

            # If roi is not None, filter predictions
            if roi is not None:
                frame_pred_bboxes = filter_roi(bboxes=frame_pred_bboxes, roi_dist=roi, th=roi_th)

            if initialize:
                for box in frame_pred_bboxes:
                    pred_ids.append(starting_id)
                    pred_centers.append((int(box[0] + box[2] / 2),
                                         int(box[3] + box[1] / 2)))
                    current_bboxes.append([box[0],
                                           box[1],
                                           box[2],
                                           box[3]])

                    tracking_predictions.append([frame_num,
                                                 starting_id,
                                                 int(box[0]),
                                                 int(box[1]),
                                                 int(box[2] - box[0]),
                                                 int(box[3] - box[1]),
                                                 1])
                    starting_id += 1
                    initialize = False
            else:
                past_bboxes_np = np.zeros((len(past_bboxes), 4))
                for idx, past_box in enumerate(past_bboxes):
                    past_bboxes_np[idx, :] = [past_box[0],
                                              past_box[1],
                                              past_box[2],
                                              past_box[3]]

                # Compare each current prediction with the past detections
                for box in frame_pred_bboxes:
                    current_bbox = [box[0],
                                    box[1],
                                    box[2],
                                    box[3]]

                    ious = np.nan_to_num(iou_list(past_bboxes_np, current_bbox))

                    # If the current prediction overlaps with any of the past detections
                    if np.all(ious==np.nan):
                        id = starting_id
                        starting_id += 1

                    # If the biggest overlap is greater than a threshold
                    elif np.nanmax(ious) > 0.3:
                        # Asign the current bbox to the same track as the overlaping bbox from the previous frame
                        id = past_ids[np.nanargmax(ious)]
                        # Disable the assigned bbox from the list, to not be assigned again
                        past_bboxes_np[np.nanargmax(ious)] = [np.nan, np.nan, np.nan, np.nan]

                    # If there is no overlap
                    else:
                        id = starting_id
                        starting_id += 1

                    pred_ids.append(id)

                    pred_centers.append((int(box[0] + box[2] / 2),
                                         int(box[3] + box[1] / 2)))

                    current_bboxes.append([box[0],
                                           box[1],
                                           box[2],
                                           box[3]])

                    tracking_predictions.append([frame_num,
                                                 id,
                                                 int(box[0]),
                                                 int(box[1]),
                                                 int(box[2] - box[0]),
                                                 int(box[3] - box[1]),
                                                 1])

            # Save the current frame predictions as past frame
            past_ids = pred_ids
            past_bboxes = current_bboxes

            accumulator.update(
                gt_ids,  # Ground truth objects in this frame
                pred_ids,  # Detector hypotheses in this frame
                mm.distances.norm2squared_matrix(gt_centers, pred_centers)
                # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
            )

        # Compute the metrics
        mh = mm.metrics.create()
        summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
        print(summary)

    elif type == 'deep_sort':
        # Create the accumulator that will be updated during each frame
        accumulator = mm.MOTAccumulator(auto_id=True)

        # Create the tracker
        tracker = DeepSort(model_path='ByteTrack/yolox/deepsort_tracker/checkpoints/ckpt.t7')

        tracking_predictions = []
        # Iterate through the frames
        for img_path in tqdm(img_paths, desc=f"Tracking {img_paths[0].split('/')[-3]}"):

            frame_num = img_path.split('/')[-1].split('.')[0]

            # Obtain the Ground Truth and predictions for the current frame
            # Using the function get() to avoid crashing when there is no key with that string
            gt_annotations = ground_truth.get(frame_num, [])
            pred_annotations = predictions.get(frame_num, [])

            # Obtain the Ground Truth and predictions for the current frame
            gt_bboxes = [anno['bbox'] for anno in gt_annotations]
            pred_bboxes = [anno['bbox'] for anno in pred_annotations]
            pred_scores = [anno['confidence'] for anno in pred_annotations]
            pred_bboxes = [[box[0], box[1], box[2], box[3], score] for box, score in
                           zip(pred_bboxes, pred_scores)]  # Convert to list

            # If roi is not None, filter predictions
            if roi is not None:
                pred_bboxes = filter_roi(bboxes=pred_bboxes, roi_dist=roi, th=roi_th)

            # Obtain the Ground Truth centers and track IDs
            gt_centers = [(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in gt_bboxes]
            gt_ids = [anno['obj_id'] for anno in gt_annotations]

            outputs = np.zeros((len(pred_bboxes), 5))
            for i, item in enumerate(pred_bboxes):
                outputs[i, :] = item

            output_tracking = tracker.update(outputs, img_path)

            det_centers = []
            det_ids = []

            for t in output_tracking:
                det_centers.append((int(t[0] + t[2] / 2), int(t[1] + t[3] / 2)))
                det_ids.append(int(t[4]))
                aux = int(t[4]) + int(starting_id)
                tracking_predictions.append(
                    [frame_num, int(aux), int(t[0]), int(t[1]), int(t[2] - t[0]), int(t[3] - t[1]), 1])

            accumulator.update(
                gt_ids,  # Ground truth objects in this frame
                det_ids,  # Detector hypotheses in this frame
                mm.distances.norm2squared_matrix(gt_centers, det_centers)
                # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
            )

            # Compute the metrics
        mh = mm.metrics.create()
        summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
        print(summary)

        starting_id = max(track[1] for track in tracking_predictions)

    return tracking_predictions, summary['idf1']['acc'], starting_id


def filter_parked_cars(annotations, img_paths, var_th=25):
    """
    Filter the parked cars from the annotations. This is done by computing the variance of the bounding boxes which
    correspond to the same object ID.
    :param annotations: list of predictions bounding boxes in the format of load_annot
    :param img_paths: list of image paths corresponding the predictions
    :param var_th: variance threshold in which the bounding boxes are considered as parked cars
    :return: list of filtered predictions with the corresponding obj_id for each detection
    """
    # Create a dictionary where the keys are the track IDs and the values are the list of bboxes
    track_bboxes = {}
    for img_path in img_paths:
        frame_num = img_path.split('/')[-1].split('.')[0]

        frame_annotations = annotations.get(frame_num, [])
        for obj in frame_annotations:
            if obj['obj_id'] not in track_bboxes:
                track_bboxes[obj['obj_id']] = [compute_bbox_centroid(obj['bbox'])]
            else:
                track_bboxes[obj['obj_id']].append(compute_bbox_centroid(obj['bbox']))

    filtered_tracks = []
    for track_id, centroids in track_bboxes.items():
        if np.mean(np.std(centroids, axis=0)) > var_th:
            filtered_tracks.append(track_id)

    filtered_detections = []
    for img_path in img_paths:
        frame_num = img_path.split('/')[-1].split('.')[0]

        frame_annotations = annotations.get(frame_num, [])
        frame_filtered_annotations = []
        for anno in frame_annotations:
            if anno['obj_id'] in filtered_tracks:
                frame_filtered_annotations.append(anno)
                filtered_detections.append([frame_num, int(anno['obj_id']), int(anno['bbox'][0]), int(anno['bbox'][1]),
                                            int(anno['bbox'][2] + anno['bbox'][0]),
                                            int(anno['bbox'][3] + anno['bbox'][1]), 1])

    return filtered_detections


def _get_features(self, bbox_xywh, ori_img):
    im_crops = []
    for box in bbox_xywh:
        x1,y1,x2,y2 = self._xywh_to_xyxy(box)
        im = ori_img[y1:y2,x1:x2]
        im_crops.append(im)
    if im_crops:
        features = self.extractor(im_crops)
    else:
        features = np.array([])
    return


def plotBBoxes(img, frame=None, cam=None, saveFrames=None, **bboxes):
    """
    Plots a set of bounding boxes on an image.
    parameters:
    ----------------
    img: numpy array of one frame of the sequence
    saveFrames: if not None, saves the frame to the specified path
    bboxes: list of bounding boxes to plot. As the argument is a *args type, several sets of bboxes can be drawn on the same frame.
            i.e., if we wanted to plot the gt and the predicted bboxes, we would call the function as follows:
            plotBBoxes(img, saveFrames, gt_bbox, pred_bbox) where gt_bbox and pred_bbox are a list of the bounding boxes to plot.
    """

    COLORS = [
        (0, 0, 255),
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 0, 128),
    ]

    LETTERS = [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (255, 255, 255),
        (0, 0, 0),
        (0, 0, 0),
        (255, 255, 255),
    ]


    for idx, set_bboxes in enumerate(bboxes.values()):
        for bbox in set_bboxes:
            if len(bbox) == 5:
                cv2.rectangle(img,
                              (round(bbox[0]), round(bbox[1])),
                              (round(bbox[2]), round(bbox[3])),
                              COLORS[int(bbox[4] % len(COLORS))],
                              2,
                              )
                cv2.rectangle(img,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0] + 60), int(bbox[1] + 30)),
                              COLORS[int(bbox[4] % len(COLORS))],
                              -1)

                cv2.putText(img,
                            str(int(bbox[4])),
                            (int(bbox[0]), int(bbox[1] + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            LETTERS[int(bbox[4] % len(LETTERS))],
                            2)

            else:
                cv2.rectangle(
                    img,
                    (round(bbox[0]), round(bbox[1])),
                    (round(bbox[2]), round(bbox[3])),
                    COLORS[idx],
                    2,
                )

        if frame is not None:
            cv2.rectangle(
                img,
                (0,0),
                (120, 70),
                (0,0,0),
                -1,
            )
            cv2.rectangle(
                img,
                (4, 4),
                (116, 66),
                (255, 255, 255),
                -1,
            )

            cv2.putText(img,
                        str(frame),
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.25,
                        (0, 0, 0),
                        3)



        if cam is not None:
            cv2.putText(img,
                        str(cam),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.25,
                        (0, 0, 0),
                        3)

    if saveFrames is not None:
        cv2.imwrite(saveFrames, img)

    return img


if __name__ == "__main__":
    all_videos_to_frames("../../../data/AICity_data/train")


