import os
import torch
import numpy as np

from PIL import Image
from datetime import datetime
from tqdm import tqdm
from os.path import join
from torch.utils.data import Dataset

from utilities.dataset_utils import load_annot

class ExtractionDataset(Dataset):
    """
    Dataset class which is in charge of loading the cropped images to finetune the backbone of the model.
    """
    def __init__(self, data_path, sequences, transform=None):
        self.data_root = data_path
        self.sequences = sequences
        self.transform = transform
        self.ids = []

        self.data = []
        id_maping = {}
        id_counter = 0

        for seq in sequences:
            for cam in sorted(os.listdir(os.path.join(data_path, seq))):
                cam_gt = load_annot(join(data_path, seq, cam, 'gt'), 'gt.txt')
                for frame_num, frame_annotations in cam_gt.items():
                    frame_path = join(data_path, seq, cam, 'frames', frame_num + '.jpg')
                    for annot in frame_annotations:
                        if id_maping.get(annot['obj_id']) is None:
                            id_maping[annot['obj_id']] = id_counter
                            id_counter += 1

                        self.data.append({'path': frame_path, 'bbox': annot['bbox'], 'id': id_maping[annot['obj_id']]})
                        self.ids.append(annot['obj_id'])

        self.counter = len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index]['path']).crop(self.data[index]['bbox']).resize((224, 224))

        if self.transform is not None:
            img = self.transform(img)

        return img, self.data[index]['id']

    def __len__(self):
        return self.counter

    def num_classes(self):
        return np.unique(self.ids).shape[0]

def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch_num):
    """
    Function to train the model with one epoch
    :param loader: Dataloder, train dataloader
    :param model: object, model to train
    :param optimizer: optimizer object
    :param loss_fn: loss
    :param scaler: scaler object
    :param device: string ('cuda' or cpu')
    :param epoch_num: int number of epoch in which the model is going to be trained
    :return: float, float (accuracy, loss)
    """

    model.cuda()  # Train mode
    epoch_start = datetime.today()
    loop = tqdm(loader, desc=f'EPOCH {epoch_num} TRAIN',
                leave=False)  # Create the tqdm bar for visualizing the progress.
    iterations = loop.__len__()
    correct = 0  # accumulated correct predictions
    total_samples = 0  # accumulated total predictions
    loss_sum = 0  # accumulated loss

    # Loop to obtain the batches of images and labels
    for (data, targets) in loop:
        data = data.to(device=device)  # Batch of images to DEVICE, where the model is
        targets = targets.to(device=device)  # Batch of labels to DEVICE, where the model is

        optimizer.zero_grad()  # Initialize the gradients

        output = model(data)  # Output of the model (logits).
        loss = loss_fn(output, targets)  # Compute the loss between the output and the ground truth
        _, predictions = torch.max(output.data, 1)  # Obtain the classes with higher probability (predicted classes)

        total_samples += data.size(0)  # subtotal of the predictions
        correct += (predictions == targets).sum().item()  # subtotal of the correct predictions
        loss_sum += loss.item() * data.size(0)  # subtotal of the correct losses

        scaler.scale(loss).backward()  # compute the backward stage updating the weights of the model
        scaler.step(optimizer)  # using the Gradient Scaler
        scaler.update()

        # loop.set_postfix(acc=correct/total, loss=loss.item())  # set current accuracy and loss

        # Tensorboard: the object writer will add the batch metrics to plot in real time
    epoch_acc = correct / total_samples
    epoch_loss = loss_sum / total_samples

    epoch_end = datetime.today()
    print("\n{} epoch: {} loss: {:.3f}, acc: {:.3f}, End time: {}, Time elapsed: {}".format(
        'Train',
        epoch_num,
        epoch_loss,
        epoch_acc,
        str(epoch_end.strftime('%d/%m/%Y %H:%M:%S')),
        str(epoch_end - epoch_start).split(".")[0])
    )

    return epoch_acc, epoch_loss
