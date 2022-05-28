import os
import faiss
import torch
import wandb
import umap
import random
import math
import motmetrics as mm
import numpy as np
import torch.nn.functional as F
import matplotlib.patheffects as PathEffects
from glob import glob
from os.path import join, exists
from tqdm import tqdm
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from utilities.dataset_utils import load_annot, write_predictions
from utilities.json_loader import load_json, save_json


# --- DATASET ---
class TripletDataset(Dataset):
    def __init__(self, data_path, sequences, split, samples=5, percentile=70, transform=None):
        """
        Function to initialize the dataset for the Triplet architecture. This function will load the detections as
        if each object was a different class, i.e. the same car can be represented in different cameras.
        The function has different modes depending on the split parameter. If split=='train' the getitem function will
        return a triplet of images (anchor, positive, negative), if split=='test' it will return a single image, to be
        able to create the token for the desired detection.

        :param data_path: path to the data folder ('../AICity_data/train')
        :param sequences: [S01, S03 ...]
        :param split: 'train' or 'test'
        :param transform: transformation to apply to the images
        """
        self.data_root = data_path
        self.split = split
        self.sequences = sequences
        self.transform = transform
        self.counter = 0

        # WHEN TRAINING OR VALIDATION
        # Initialize all the tracks and its detections in the self.data dictionary. The key is the track id and the
        # value is a list of dictionaries with following structure. For each track in every cam, the detections are
        # equally spaced in the corresponding percentile.
        # {'1': [{'path': '../AICity_data/train/S01/c001/0010.jpg',
        #         'bbox': [x1, y1, x2, y2],
        #         'cam': 'c001'},
        #         {'path': '../AICity_data/train/S01/c001/0011.jpg',
        #         'bbox': [x1, y1, x2, y2]}],
        #         'cam': 'c001'},
        #                 ...
        #       ]
        #  '2': [{'path': '../AICity_data/train/S01/c001/0012.jpg',
        #         'bbox': [x1, y1, x2, y2]},
        #         'cam': 'c001'},
        #    ...
        #       ]
        #  }

        if split == "train":

            self.data = {}

            self.cam_data = {}

            # Iterate through the sequences
            for seq in sequences:
                # Iterate through the cameras
                for cam in sorted(os.listdir(os.path.join(data_path, seq))):
                    cam_gt = load_annot(join(data_path, seq, cam, 'gt'), 'gt.txt')
                    self.cam_data[cam] = {}
                    for frame_num, frame_annotations in cam_gt.items():
                        frame_path = join(data_path, seq, cam, 'frames', frame_num + '.jpg')
                        for annot in frame_annotations:

                            if self.cam_data[cam].get(annot['obj_id']) is None:
                                self.cam_data[cam][annot['obj_id']] = [{'path': frame_path,
                                                                        'bbox': annot['bbox'],
                                                                        'cam': cam}]
                            else:
                                self.cam_data[cam][annot['obj_id']].append({'path': frame_path,
                                                                            'bbox': annot['bbox'],
                                                                            'cam': cam})

            for cam_name, cam_annotations in self.cam_data.items():
                for obj_id, obj_annotations in cam_annotations.items():
                    if len(obj_annotations) > samples:
                        boxes_area = [(a['bbox'][3] - a['bbox'][1]) * (a['bbox'][2] - a['bbox'][0]) for a in
                                      obj_annotations]

                        area_percentile = np.percentile(boxes_area, percentile)

                        filtered_annot = [a for a in obj_annotations if ((a['bbox'][2] - a['bbox'][0]) *
                                                                         (a['bbox'][3] - a['bbox'][
                                                                             1])) >= area_percentile]

                        if len(filtered_annot) > samples:
                            interval = math.ceil(len(filtered_annot) / samples)
                            sampled_annot = [filtered_annot[i] for i in range(0, len(filtered_annot), interval)]
                        else:
                            sampled_annot = filtered_annot

                        if self.data.get(obj_id) is None:
                            self.data[obj_id] = [a for a in sampled_annot]

                        else:
                            for a in sampled_annot:
                                self.data[obj_id].append(a)
                        self.counter += len(sampled_annot)
                    else:
                        if self.data.get(obj_id) is None:
                            self.data[obj_id] = [a for a in obj_annotations]

                        else:
                            for a in obj_annotations:
                                self.data[obj_id].append(a)
                        self.counter += len(obj_annotations)

        # WHEN TESTING
        # Initialize the self.data with a list of dictionaries of all the detections of the test set.
        # In order to be able to create the token for the desired detection, the data will be a list of dictionaries
        # with the following structure:
        # [{'path': '../AICity_data/test/S01/c001/0012.jpg',
        #   'bbox': [x1, y1, x2, y2],
        #   'id': 1,
        #   'cam': 'c001'},
        #  {'path': '../AICity_data/test/S01/c001/0013.jpg',
        #   'bbox': [x1, y1, x2, y2]},
        #   'id': 1,
        #   'cam': 'c001'},
        #  ...
        #  ]
        elif split == "test":
            self.data = []
            self.cam_data = {}
            # Iterate through the sequences
            for seq in sequences:
                # Iterate through the cameras
                for cam in sorted(os.listdir(os.path.join(data_path, seq))):
                    cam_gt = load_annot(join(data_path, seq, cam, 'gt'), 'gt.txt')
                    self.cam_data[cam] = {}
                    for frame_num, frame_annotations in cam_gt.items():
                        frame_path = join(data_path, seq, cam, 'frames', frame_num + '.jpg')
                        for annot in frame_annotations:

                            if self.cam_data[cam].get(annot['obj_id']) is None:
                                self.cam_data[cam][annot['obj_id']] = [{'path': frame_path,
                                                                        'bbox': annot['bbox'],
                                                                        'cam': cam,
                                                                        'id': annot['obj_id']}]
                            else:
                                self.cam_data[cam][annot['obj_id']].append({'path': frame_path,
                                                                            'bbox': annot['bbox'],
                                                                            'cam': cam,
                                                                            'id': annot['obj_id']})
            for cam_name, cam_annotations in self.cam_data.items():
                for obj_id, obj_annotations in cam_annotations.items():

                    if len(obj_annotations) > samples:
                        boxes_area = [(a['bbox'][3] - a['bbox'][1]) * (a['bbox'][2] - a['bbox'][0]) for a in
                                      obj_annotations]

                        area_percentile = np.percentile(boxes_area, percentile)

                        filtered_annot = [a for a in obj_annotations if ((a['bbox'][2] - a['bbox'][0]) *
                                                                         (a['bbox'][3] - a['bbox'][
                                                                             1])) > area_percentile]

                        if len(filtered_annot) > samples:
                            interval = math.ceil(len(filtered_annot) / samples)
                            sampled_annot = [filtered_annot[i] for i in range(0, len(filtered_annot), interval)]
                        else:
                            sampled_annot = filtered_annot

                        for a in sampled_annot:
                            self.data.append(a)

                    else:
                        for a in obj_annotations:
                            self.data.append(a)

            self.counter = len(self.data)

        else:
            print('WRONG SPLIT PARAMETER...')

    def __getitem__(self, index):

        if self.split == 'train':
            counter = 0
            max_id = max(list(self.data.keys()))

            for id in range(1, max_id + 1):
                if self.data.get(id) is not None:
                    if index >= (len(self.data[id]) + counter):
                        counter += len(self.data[id])
                    else:
                        anchor_id = id
                        anchor_img = self.data[id][index - counter]
                        break

            # Get the positive image
            positive_img = random.choice(self.data[anchor_id])

            # Get negative image
            filtered_list = [x for x in list(self.data.keys()) if x != anchor_id]
            negative_id = random.choice(filtered_list)
            negative_img = random.choice(self.data[negative_id])

            anchor_img = Image.open(anchor_img['path']).crop(anchor_img['bbox']).resize((224, 224))
            positive_img = Image.open(positive_img['path']).crop(positive_img['bbox']).resize((224, 224))
            negative_img = Image.open(negative_img['path']).crop(negative_img['bbox']).resize((224, 224))

            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return (anchor_img, positive_img, negative_img), []

        elif self.split == 'test':
            img = Image.open(self.data[index]['path']).crop(self.data[index]['bbox']).resize((224, 224))

            if self.transform is not None:
                img = self.transform(img)

            return img, self.data[index]['id']

        else:
            print('WRONG SPLIT PARAMETER...')

    def __len__(self):
        return self.counter

# --- MODELS ---

class EmbeddingNet(nn.Module):
    def __init__(self, model):
        super(EmbeddingNet, self).__init__()

        self.base_resnet = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, x):
        return self.base_resnet(x).squeeze()

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


# --- TRAINING ---
def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, output_path,
        model_id,
        metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    wandb.init(project="M6-week5", entity='celulaeucariota', name=model_id)

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics, log_interval)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        PATH = join(output_path, model_id + '.pth')
        torch.save(model.state_dict(), PATH)
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    """
    Train for one epoch on the training set.
    :param train_loader:    Train data loader
    :param model:
    :param loss_fn:
    :param optimizer:
    :param cuda:
    :param log_interval:
    :param metrics:
    :return:
    """
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics, log_interval):
    """
    Evaluate model on test set
    :param val_loader:
    :param model:
    :param loss_fn:
    :param cuda:
    :param metrics:
    :param log_interval:
    :return:
    """
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == 0:
                message = 'Val: [{}/{} ({:.0f}%)]'.format(batch_idx * len(data[0]), len(val_loader.dataset),
                                                          100. * batch_idx / len(val_loader))
                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses = []

    return val_loss, metrics


# --- LOSS FUNCTION ---
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class Matcher():
    """
    Matcher class. This class is used to match the car embedding of different cameras.
    """
    def __init__(self, data_path, sequence, samples, percentile, model, model_id, type, transform):

        self.data_path = data_path

        if isinstance(sequence, str):
            self.sequence = sequence
        else:
            self.sequence = sequence[0]

        self.sequence = sequence
        self.model = model.cuda()
        self.transform = transform

        self.seq_embeddings = {}

        self.train_sequences = ["S01", "S03", "S04"]
        self.train_sequences.remove(self.sequence)

        self.type = type

        json_path = join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'dataset', f'{self.type}_embeddings_{model_id}.json')
        if exists(json_path):
            print(f"Loading the embeddings of {model_id}...")
            print(f'If you want to create new embeddings for that model, delete {json_path}')
            (self.cam_data, self.seq_embeddings) = load_json(json_path)

        else:
            self.cam_data = {}

            for cam in tqdm(sorted(os.listdir(join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'mtsc_' + self.type))),
                            desc="Creating the embeddings"):
                tracking_pred = load_annot(join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'mtsc_' + self.type), cam)
                self.cam_data[cam.split('.')[0]] = {}
                for frame_num, frame_annotations in tracking_pred.items():
                    frame_path = join(data_path, self.sequence, cam.split('.')[0], 'frames', frame_num + '.jpg')
                    for annot in frame_annotations:

                        if self.cam_data[cam.split('.')[0]].get(annot['obj_id']) is None:
                            self.cam_data[cam.split('.')[0]][annot['obj_id']] = [{'path': frame_path,
                                                               'bbox': annot['bbox'],
                                                               'cam': cam.split('.')[0],
                                                               'id': annot['obj_id']}]
                        else:
                            self.cam_data[cam.split('.')[0]][annot['obj_id']].append({'path': frame_path,
                                                                   'bbox': annot['bbox'],
                                                                   'cam': cam.split('.')[0],
                                                                   'id': annot['obj_id']})

            for cam_name, cam_annotations in self.cam_data.items():
                self.seq_embeddings[cam_name] = {'embeddings': [], 'labels': []}
                for obj_id, obj_annotations in cam_annotations.items():

                    if len(obj_annotations) > samples:
                        boxes_area = [(a['bbox'][3] - a['bbox'][1]) * (a['bbox'][2] - a['bbox'][0]) for a in
                                      obj_annotations]
                        area_percentile = np.percentile(boxes_area, percentile)

                        filtered_annot = [a for a in obj_annotations if ((a['bbox'][2] - a['bbox'][0]) *
                                                                         (a['bbox'][3] - a['bbox'][
                                                                             1])) > area_percentile]

                        if len(filtered_annot) > samples:
                            interval = math.ceil(len(filtered_annot) / samples)
                            sampled_annot = [filtered_annot[i] for i in range(0, len(filtered_annot), interval)]
                        else:
                            sampled_annot = filtered_annot

                        for annot in sampled_annot:
                            img = Image.open(annot['path']).crop(annot['bbox']).resize((224, 224))
                            img = self.transform(img)

                            if not 'triplet' in model_id:
                                emb = self.model(img.unsqueeze(0).cuda()).detach().cpu().numpy().tolist()[0]
                            else:
                                emb =self.model.get_embedding(img.unsqueeze(0).cuda()).detach().cpu().numpy().tolist()
                            self.seq_embeddings[cam_name]['embeddings'].append(emb)
                            self.seq_embeddings[cam_name]['labels'].append(annot['id'])


                    elif (len(obj_annotations) < samples) and (len(obj_annotations) >= 3):
                        sampled_annot = obj_annotations

                        for annot in sampled_annot:
                            img = Image.open(annot['path']).crop(annot['bbox']).resize((224, 224))
                            img = self.transform(img)

                            if not 'triplet' in model_id:
                                emb = self.model(img.unsqueeze(0).cuda()).detach().cpu().numpy().tolist()[0]
                            else:
                                emb = self.model.get_embedding(img.unsqueeze(0).cuda()).detach().cpu().numpy().tolist()

                            self.seq_embeddings[cam_name]['embeddings'].append(emb)
                            self.seq_embeddings[cam_name]['labels'].append(annot['id'])

            save_json(json_path, (self.cam_data, self.seq_embeddings))

    def match(self, embeddings1, labels1, embeddings2, labels2, distance_th=1.3, n_neighbors=3):
        """
        Match embeddings from two different cameras.
        :param embeddings1:
        :param labels1:
        :param embeddings2:
        :param labels2:
        :return:
        """
        indices1 = {}
        distances1 = {}

        # Create faiss and add embeddings2
        index = faiss.IndexFlatL2(2048)
        index.add(np.array(embeddings2).astype(np.float32))

        unique_ids = np.unique(labels1)
        for id in unique_ids:
            id_embeddings = [embeddings1[i] for i, x in enumerate(labels1) if x == id]
            id_embeddings = np.array(id_embeddings).astype(np.float32)
            D, I = index.search(id_embeddings, n_neighbors)
            track_ind = []
            for indices in I:
                track_ind.append([labels2[i] for i in indices])
            distances1[id] = [distances.tolist() for distances in D]
            indices1[id] = track_ind

        indices2 = {}
        distances2 = {}

        # Create faiss and add embeddings2
        index = faiss.IndexFlatL2(2048)
        index.add(np.array(embeddings1).astype(np.float32))

        unique_ids = np.unique(labels2)
        for id in unique_ids:
            id_embeddings = [embeddings2[i] for i, x in enumerate(labels2) if x == id]
            id_embeddings = np.array(id_embeddings).astype(np.float32)
            D, I = index.search(id_embeddings, n_neighbors)
            track_ind = []
            for indices in I:
                track_ind.append([labels1[i] for i in indices])
            distances2[id] = [distances.tolist() for distances in D]
            indices2[id] = track_ind

        reids_matches = []
        for (id, indices), (_, distances) in zip(indices1.items(), distances1.items()):
            indices = [i for sub_ind, sub_dis in zip(indices, distances) for i, d in zip(sub_ind, sub_dis) if d < distance_th]

            if len(indices) > 0:
                match_id = max(set(indices), key=indices.count)

                cross_indices = indices2[match_id]
                cross_distances = distances2[match_id]

                indices = [i for sub_ind, sub_dis in zip(cross_indices, cross_distances) for i, d in zip(sub_ind, sub_dis) if d < distance_th]

                if len(indices) > 0:
                    cross_match_id = max(set(indices), key=indices.count)
                    if cross_match_id == id:
                        reids_matches.append((id, match_id))

        return reids_matches

    def match_all(self, distance_th=1.3, n_neighbors=3):
        """
        Match all embeddings from all cameras in a sequential manner:

        c010 --|
               |re-id|--|
        c011 --|        |re-id|--|
        c012 -----------|        |re-id|--|
        c013 --------------------|
                     ...

        :param distance_th:
        :param n_neighbors:
        :return:
        """
        cam_names = list(self.cam_data.keys())

        accumulative_emb = self.seq_embeddings[cam_names[0]]['embeddings']
        accumulative_labels = self.seq_embeddings[cam_names[0]]['labels']
        for i in range(1, len(cam_names)-1):
            embeddings2 = self.seq_embeddings[cam_names[i]]['embeddings']
            labels2 = self.seq_embeddings[cam_names[i]]['labels']

            reids = self.match(embeddings1=accumulative_emb,
                               labels1=accumulative_labels,
                               embeddings2=embeddings2,
                               labels2=labels2,
                               distance_th=distance_th,
                               n_neighbors=n_neighbors)

            for (id, match_id) in reids:
                self.cam_data[cam_names[i]][str(id)] = self.cam_data[cam_names[i]].pop(str(match_id))
                lab = self.seq_embeddings[cam_names[i]]['labels']
                self.seq_embeddings[cam_names[i]]['labels'] = [int(id) if x==match_id else x for x in lab]

            accumulative_emb = accumulative_emb + embeddings2
            accumulative_labels = accumulative_labels + self.seq_embeddings[cam_names[i]]['labels']

        os.makedirs(join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'mtmc_' + self.type), exist_ok=True)

        for cam_name, annotations in self.cam_data.items():
            cam_annot = []
            for track_id, annot in annotations.items():
                for a in annot:
                    frame_num = a['path'].split('/')[-1].split('.')[0]
                    cam_annot.append([frame_num,
                                     track_id,
                                     a['bbox'][0],
                                     a['bbox'][1],
                                     a['bbox'][2]-a['bbox'][0],
                                     a['bbox'][3]-a['bbox'][1],
                                     1])

            write_predictions(join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'mtmc_' + self.type, cam_name + '.txt'), cam_annot)

    def eval_mtmc(self):
        """
        Evaluate the tracking results on the entire sequence.
        :return:
        """
        accumulator = mm.MOTAccumulator(auto_id=True)
        for cam in sorted(os.listdir(join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'mtmc_' + self.type))):

            ground_truth = load_annot(join(self.data_path, self.sequence, cam.split('.')[0], 'gt'), 'gt.txt')
            predictions = load_annot(join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'mtmc_' + self.type), cam)

            for img_path in sorted(glob(join(self.data_path, self.sequence, cam.split('.')[0], 'frames', '*.jpg'))):
                frame_num = img_path.split('/')[-1].split('.')[0]

                # Obtain the Ground Truth and predictions for the current frame
                # Using the function get() to avoid crashing when there is no key with that string
                gt_annotations = ground_truth.get(frame_num, [])
                pred_annotations = predictions.get(frame_num, [])

                # Obtain both the Ground Truth and predictions centers
                gt_centers = [(a['bbox'][0] + a['bbox'][2] / 2, a['bbox'][1] + a['bbox'][3] / 2) for a in gt_annotations]
                pred_centers = [(a['bbox'][0] + a['bbox'][2] / 2, a['bbox'][1] + a['bbox'][3] / 2) for a in pred_annotations]

                # Obtain both the Ground Truth ids and predictions IDs
                gt_ids = [a['obj_id'] for a in gt_annotations]
                pred_ids = [a['obj_id'] for a in pred_annotations]

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

    # Draw UMAP
    def draw_umap(self, num_ids):
        """
        Draw and save UMAP of the embeddings
        :param model: model object
        :param dataset:
        :param cameras:
        :param save_path:
        :return:
        """

        # Generate radom color pallette
        COLORS = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in
                                 range(6)]) for i in range(num_ids)]

        features = []
        labels = []
        for seq_emb in self.seq_embeddings.values():
            for embed, lab in zip(seq_emb['embeddings'], seq_emb['labels']):
                features.append(embed)
                labels.append(lab)

        # Generate UMAP
        embeddings = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(np.array(features))

        # Plot UMAP
        unique_ids = np.unique(labels)

        unique_ids = random.sample(unique_ids.tolist(), num_ids)

        for idx, id in enumerate(unique_ids):
            id_features = [embeddings[i] for i, x in enumerate(labels) if x == id]
            plt.scatter(
                [x[0] for x in id_features],
                [x[1] for x in id_features],
                c=COLORS[idx],
                label=f'{id}',
            )
            xtext, ytext = np.median(id_features, axis=0)
            txt = plt.text(xtext, ytext, f'{id}', fontsize=10, fontweight='bold')
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=3, foreground="w"),
                PathEffects.Normal()])

        plt.legend(loc='best')
        plt.title(f'2D UMAP representation of {num_ids} cars of sequence {self.sequence}')

        os.makedirs(join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'figures'), exist_ok=True)
        plt.savefig(fname=join('data', 'fasterrcnn', '-'.join(self.train_sequences), 'figures', 'umap.png'))
