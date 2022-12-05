import os
from tqdm import tqdm
import numpy as np

def calculate_weigths_labels(path, dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = path
    np.save(classes_weights_path, ret)

    return ret


def calculate_weigths_labels_new(path, dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    # epsilon = 0.2           # 0.1, 1e-2, 1e-4
    for frequency in z:
        # class_weight = np.log(total_frequency / (2*frequency))
        # class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        # class_weight = 1 / (np.log(1 + epsilon + (frequency / total_frequency)))
        class_weight = frequency / total_frequency
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = path
    np.save(classes_weights_path, ret)

    return ret