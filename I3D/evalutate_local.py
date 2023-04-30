import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import videotransforms
import numpy as np
from I3D import InceptionI3d
from dataset import WLASLDataset as Dataset


def load_labels(fname):
    labels = dict()
    with open(fname) as f:
        for line in f.readlines():
            splt = line.split()
            labels[splt[1]] = splt[0]
    return labels


def main(root, data_file, weights, confusion_matrix=None):
    if confusion_matrix is not None:
        labels = load_labels('wlasl_class_list.txt')

        confusion_matrix = np.load(confusion_matrix)
        true_pos = np.diag(confusion_matrix)
        precision = true_pos / np.sum(confusion_matrix, axis=1)
        precision = np.nan_to_num(precision)
        recall = true_pos / np.sum(confusion_matrix, axis=0)
        recall = np.nan_to_num(recall)
        accuracy = np.sum(true_pos) / confusion_matrix.sum()
        epsilon = 1e-308
        with open('train_metrics.txt', 'w') as f:
            for i, cls in enumerate(labels):
                cls_precision = precision[i]
                cls_recall = recall[i]
                cls_f1 = 2 * ((cls_precision * cls_recall) / (cls_precision + cls_recall + epsilon))
                if cls_precision != 0.0 and cls_recall != 0.0 and cls_f1 != 0.0:
                    f.write(f"Metrics for class {cls}:\n\tPrecision: {cls_precision}\n\tRecall: {cls_recall}\n\tF1: {cls_f1}\n\n")
            f.write(f'Total Class Accuracy: {accuracy}')
    else:
        # setup dataset
        test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

        dataset = Dataset(data_file, 'test', root, test_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

        dataloaders = {'test': dataloader}

        # setup the model
        i3d = InceptionI3d(2000, in_channels=3)
        num_classes = dataset.num_classes
        i3d.load_state_dict(torch.load(weights))

        # device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f'Using GPU: {device}')

        i3d.to(device)
        i3d = nn.DataParallel(i3d)
        i3d.eval()

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for data in dataloaders['test']:
            # inputs, labels, vid, src = data
            inputs, labels, vid = data

            inputs = inputs.to(device)
            t = inputs.size(2)
            labels = labels.to(device)

            per_frame_logits = i3d(inputs)
            per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')

            predictions = torch.max(per_frame_logits, dim=2)[0]
            gt = torch.max(labels, dim=2)[0]

            for i in range(per_frame_logits.shape[0]):
                confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

        print(float(np.trace(confusion_matrix)) / np.sum(confusion_matrix))
        np.save("confusion_matrix.npy", confusion_matrix)


if __name__ == '__main__':
    # Test accuracy 0.14562722595963593
    root = '../data'
    data_file = 'train_test_val.json'
    main(root=root, data_file=data_file, weights='checkpoints/nslt_2000_044946_0.142461.pt',
         confusion_matrix='train_confusion_matrix.npy')
