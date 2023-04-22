from dataset import WLASLSegmentDataset
from model.model import CNN3D

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
data_set = WLASLSegmentDataset('', '10_frame_segments.csv', labels_map)

conv_layers = [(10, 32, 3, 2, 1, 2),
               (32, 64, 2, 1, 1, 2),
               (64, 64, 2, 1, 1, 2),
               (64, 128, 2, 1, 1, 2),
               (128, 256, 2, 1, 1, 2),
               (256, 256, 2, 1, 1, 2)]
fc_layers = [5096, 2048]

model = CNN3D(2000, conv_layers, fc_layers, 0.20, save_fqp='10_frame_segments/model.pt')
model.float()

learning_rate = 0.001
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
model.set_optimizer(optimizer)

num_epochs = 5
loss_list = []
accuracy_list = []

# Have to have batch size of 1 because each training sample can have a different number of segments. The data loader
# does not allow this.
train_loader = DataLoader(data_set, batch_size=1, shuffle=False)

start = time.process_time()
model.train_model(train_loader, num_epochs)
end = time.process_time()
print(f'Time to train: {datetime.timedelta(seconds=end-start)}')
