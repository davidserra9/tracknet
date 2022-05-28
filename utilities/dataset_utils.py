import glob
import os
import numpy as np
import xml.etree.ElementTree as ET
from os.path import join
from tqdm import tqdm
from PIL import Image
import subprocess
from os.path import exists
import yaml


def update_data(annot, frame_id, xmin, ymin, xmax, ymax, conf, obj_id=0, parked=False):
    """
    Updates the annotations dict with by adding the desired data to it
    :param annot: annotation dict
    :param frame_id: id of the framed added
    :param xmin: min position on the x axis of the bbox
    :param ymin: min position on the y axis of the bbox
    :param xmax: max position on the x axis of the bbox
    :param ymax: max position on the y axis of the bbox
    :param conf: confidence
    :return: the updated dictionary
    """

    frame_name = '%04d' % int(frame_id)
    obj_info = dict(
        name='car',
        obj_id=obj_id,
        bbox=list(map(float, [xmin, ymin, xmax, ymax])),
        confidence=float(conf),
        parked=parked
    )

    if frame_name not in annot.keys():
        annot.update({frame_name: [obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot


def load_text(text_dir, text_name):
    """
    Parses an annotations TXT file
    :param xml_dir: dir where the file is stored
    :param xml_name: name of the file to parse
    :return: a dictionary with the data parsed
    """
    with open(join(text_dir, text_name), 'r') as f:
        txt = f.readlines()

    annot = {}
    for frame in txt:
        frame_id, bb_id, xmin, ymin, width, height, conf, _, _, _ = list(map(float, (frame.split('\n')[0]).split(',')))
        update_data(annot, frame_id - 1, xmin, ymin, xmin + width, ymin + height, conf, int(bb_id))
    return annot


def load_xml(xml_dir, xml_name, ignore_parked=True):
    """
    Parses an annotations XML file
    :param xml_dir: dir where the file is stored
    :param xml_name: name of the file to parse
    :return: a dictionary with the data parsed
    """
    tree = ET.parse(join(xml_dir, xml_name))
    root = tree.getroot()
    annot = {}

    for child in root:
        if child.tag in 'track':
            if child.attrib['label'] not in 'car':
                continue
            obj_id = int(child.attrib['id'])
            for bbox in child.getchildren():
                '''if bbox.getchildren()[0].text in 'true':
                    continue'''
                frame_id, xmin, ymin, xmax, ymax, _, _, _ = list(map(float, ([v for k, v in bbox.attrib.items()])))
                update_data(annot, int(frame_id) + 1, xmin, ymin, xmax, ymax, 1., obj_id)

    return annot

def load_annot(annot_dir, name, ignore_parked=True):
    """
    Loads annotations in XML format or TXT
    :param annot_dir: dir containing the annotations
    :param name: name of the file to load
    :return: the loaded annotations
    """
    if name.endswith('txt'):
        annot = load_text(annot_dir, name)
    elif name.endswith('xml'):
        annot = load_xml(annot_dir, name, ignore_parked)
    else:
        assert 'Not supported annotation format ' + name.split('.')[-1]

    return annot

def gt_multi_txt(path, bboxes):
    """
        Convert bboxes in AICity format to YOLOv3 utralytics format.
        :param bboxes: list of bboxes in format (xmin, ymin, xmax, ymax)
    """

    W, H = Image.open(path).size

    lines_out = []
    for obj_info in bboxes:
        label = 0  # obj_info['name']
        xmin, ymin, xmax, ymax = obj_info['bbox']

        cx = '%.3f' % np.clip(((xmax + xmin) / 2) / W, 0, 1)
        cy = '%.3f' % np.clip(((ymax + ymin) / 2) / H, 0, 1)
        w = '%.3f' % np.clip((xmax - xmin) / W, 0, 1)
        h = '%.3f' % np.clip((ymax - ymin) / H, 0, 1)

        lines_out.append(' '.join([str(label), cx, cy, w, h, '\n']))

    return lines_out


def to_yolo(data, gt_bboxes):
    """
        Convert AICity data format to YOLOv3 utralytics format.
        :param data: paths to train and validation files
        :param gt_bboxes: dict of ground truth detections
        :return split_txt: dict with path to files for every split (train, val, test)
    """
    splits_txt = {}
    for split, split_data in data.items():
        files = []
        for cam, paths in tqdm(split_data.items(), 'Preparing ' + split + ' data for YOLOv5'):
            txts = glob.glob(os.path.dirname(paths[0]) + '/*.txt')

            for path in paths:
                # Add path
                files.append(path + '\n')

                if len(paths) == len(txts):
                    continue

                # Convert to yolov3 format
                frame_id = os.path.basename(path).split('.')[0]
                lines_out = gt_multi_txt(path, gt_bboxes[cam].get(frame_id, []))

                # Write/save files
                file_out = open(path.replace('jpg', 'txt'), 'w')
                file_out.writelines(lines_out)

        splits_txt.update({split: files})

    return splits_txt

def write_yaml_file(yaml_dict, yaml_file):
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_dict, f)

    if exists(yaml_file):
        print('YAML file ' + yaml_file + ' written successfully!')

def list_to_dict(list_data):
    """
    Convert list of data to dict.
    The input data should be a list of list with the annotations in the following format:

    INPUT:
    [[frame_id, obj_id, xmin, ymin, xmax, ymax, conf],
     [frame_id, obj_id, xmin, ymin, xmax, ymax, conf],
     [frame_id, obj_id, xmin, ymin, xmax, ymax, conf],
     ...]

    The output data is a dictionary with the following format:

    OUTPUT:
    {frame_id: [{'name': 'car',
                 'obj_id': obj_id,
                 'bbox': [xmin, ymin, xmax, ymax],
                 'confidence': conf},
                {'name': 'car',
                 'obj_id': obj_id,
                 'bbox': [xmin, ymin, xmax, ymax],
                 'confidence': conf}
                    ...]
     frame_id: [{'name': 'car',
                 'obj_id': obj_id,
                 'bbox': [xmin, ymin, xmax, ymax],
                 'confidence': conf},
                {'name': 'car',
                 'obj_id': obj_id,
                 'bbox': [xmin, ymin, xmax, ymax],
                 'confidence': conf}
                    ...]
     ...
    }

    :param list_data: list of data
    :return: dict of data
    """
    data = {}
    for item in list_data:
        obj = {'name': 'car',
               'obj_id': item[1],
               'bbox': [item[2], item[3], item[4], item[5]],
                'confidence': item[6]}

        if data.get(item[0]) is None:
            data[item[0]] = [obj]
        else: data[item[0]].append(obj)
    return data

def write_predictions(path, annotations):
    """
    Write the predictions in the .txt file.
    :param path: txt file path where the predictions will be written. You must ensure that the subfolders exist
    :param annotations: list of list with all predicions. It has to have the following format:
    [[frame_id, id, xmin, ymin, xmax, ymax, confidence],
     [frame_id, id, xmin, ymin, xmax, ymax, confidence],
        ...
    ]
    """
    with open(path, 'w') as f:
        for anno in annotations:
            f.write(f'{int(anno[0])},{int(anno[1])},{int(anno[2])},{int(anno[3])},{int(anno[4])},{int(anno[5])},{anno[6]},-1,-1,-1\n')


def get_weights(model):
    os.makedirs('data/weights/', exist_ok=True)
    if model.endswith('.pt') or model.endswith('.pkl'):
        model_path = model
    else:
        model_path = 'data/weights/' + model + '.pt'

    if not exists(model_path):
        subprocess.call(['sh', './data/scripts/get_' + model + '.sh'])

    return model_path