from dataset import WLASLSegmentDataset, PaddedWLASLDataset, WLASLDataset
from model.model import Segmented3DCNN, Padded3DCNN

import csv
import datetime
import time
import torch
from torch.utils.data import DataLoader


def load_labels(fname):
    labels = {}
    with open(fname, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, label in enumerate(next(reader)):
            labels[label] = i
    return labels


labels_map = load_labels('labels.csv')
segment = False
pad = True
dataset = None
save_fqp = None
model = None

conv_layers = [(3, 32, 3, 2, 1, 2),
               (32, 64, 3, 2, 1, 2),
               (64, 128, 3, 2, 1, 2),
               (128, 128, 2, 1, 1, 2),
               ]
fc_layers = [3500, 2000]

if segment:
    save_fqp = './10_frame_segments'
    model = Segmented3DCNN(2000, conv_layers, fc_layers, 0.20, 2, save_fqp=save_fqp)
    dataset = WLASLSegmentDataset('', '10_frame_segments.csv', labels_map)
elif pad:
    save_fqp = './padded_videos/'
    model = Padded3DCNN(2000, conv_layers, fc_layers, 0.1, 2, save_fqp=save_fqp)
    dataset = PaddedWLASLDataset('', 'padded_videos_train.csv', labels_map)
else:
    save_fqp = './resized_videos/'
    model = Padded3DCNN(2000, conv_layers, fc_layers, 0.1, 2, save_fqp=save_fqp)
    dataset = WLASLDataset('', 'videos_train.csv', labels_map)

num_epochs = 5
learning_rate = 1e-8
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model.set_optimizer(optimizer)

# Have to have batch size of 1 if using segments because each training sample can have a different number of segments.
# The data loader does not allow this.
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

start = time.process_time()
model.train_model(train_loader, num_epochs, learning_rate)
end = time.process_time()
print(f'Time to train: {datetime.timedelta(seconds=end - start)}')
