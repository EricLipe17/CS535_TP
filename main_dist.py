import argparse
import csv
import numpy as np
import os
import pandas as pd
import random
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class PaddedWLASLDataset(Dataset):
    def __init__(self, data_dir, csv_file, labels_map=None):
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file, header=None)
        self.data_dir = data_dir
        self.labels_map = labels_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_file = os.path.abspath(os.path.join(self.data_dir, self.df.iloc[idx, 0]))
        frames, _, _ = torchvision.io.read_video(video_file)

        label = self.df.iloc[idx, 1]
        label_index = self.labels_map[label]
        label_tensor = torch.zeros((len(self.labels_map),))
        label_tensor[label_index] = 1

        frames = frames.reshape((3, 255, 640, 640)) / 255.

        return label_tensor, frames


class CNN3D(torch.nn.Module):
    def __init__(self, num_classes, conv_layers, fc_layers, p_drop):
        super(CNN3D, self).__init__()
        self.num_classes = num_classes
        self.num_conv_layers = len(conv_layers)
        self.num_fc_layers = len(fc_layers)

        self.layers = torch.nn.ParameterList()
        for conv_layer in conv_layers:
            self.layers.append(self._build_conv_layer(conv_layer[0], conv_layer[1], conv_layer[2], conv_layer[3],
                                                      conv_layer[4], conv_layer[5]))

        # TODO: Figure out how to calculate this from parameters
        prev_layer_size = 6400  # 2**3*conv_layers[-1][1]
        for fc_layer in fc_layers:
            self.layers.append(torch.nn.Linear(prev_layer_size, fc_layer))
            self.layers.append(torch.nn.LeakyReLU())
            prev_layer_size = fc_layer

        self.layers.append(torch.nn.InstanceNorm1d(fc_layers[-1]))

        self.layers.append(torch.nn.Dropout(p_drop))

        self.layers.append(torch.nn.Softmax(dim=1))

    @staticmethod
    def _build_conv_layer(in_channels, out_channels, conv_kernel_size, conv_stride, conv_padding, pool_kernel_size):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, conv_kernel_size, conv_stride, conv_padding),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(pool_kernel_size)
        )
        return layer

    def forward(self, frames):
        x = self.layers[0](frames)
        prev_was_conv = False
        for j in range(1, len(self.layers)):
            layer = self.layers[j]
            if type(layer) is torch.nn.Sequential:
                prev_was_conv = True
            if prev_was_conv and type(layer) is torch.nn.Linear:
                x = x.flatten()
                prev_was_conv = False
            if type(layer) is torch.nn.InstanceNorm1d:
                x = x.unsqueeze(0)
            x = layer(x)
        return x


def load_labels(fname):
    labels = {}
    with open(fname, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, label in enumerate(next(reader)):
            labels[label] = i
    return labels


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    num_epochs_default = 20
    batch_size_default = 1
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "ddp_model.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.", default=False)
    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")

    labels_map = load_labels('labels.csv')
    train_dataset = PaddedWLASLDataset('', 'padded_videos_train.csv', labels_map)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    device = torch.device("cuda:{}".format(local_rank))

    loss_func = torch.nn.CrossEntropyLoss()
    loss_func.to(device)

    conv_layers = [(3, 32, 3, 2, 1, 2),
                   (32, 64, 3, 2, 1, 2),
                   (64, 128, 3, 2, 1, 2),
                   (128, 128, 2, 1, 1, 2),
                   ]
    fc_layers = [3500, 2000]
    model = CNN3D(2000, conv_layers, fc_layers, 0.1)
    model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate)

    # We only save the model who uses rank 0 and device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume is True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    for epoch in range(num_epochs):
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        ddp_model.train()

        for j, (label, frames) in enumerate(train_loader):
            # Move to device
            frames = frames.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            prediction = ddp_model(frames)

            # Calculate softmax and cross entropy loss
            label = label.to(device)
            prediction = prediction.to(device)
            loss = loss_func(prediction, label)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            if j % 50 == 0 or j == 0 or j == len(train_loader) - 1:
                if local_rank == 0:
                    print(f'Epoch: {epoch}, Iteration: {j}, Loss: {loss.data.item()}')
                    torch.save(ddp_model.state_dict(), model_filepath)
