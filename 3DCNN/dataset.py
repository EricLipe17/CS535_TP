import cv2
import json
import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torch.utils.data import Dataset


class WLASLSegmentDataset(Dataset):
    def __init__(self, data_dir, csv_file, labels_map=None):
        self.csv_file = csv_file
        self.num_frames_per_segment = int(self.csv_file.split('_')[0])
        col_names = [i for i in range(self.determine_max_num_columns())]
        self.df = pd.read_csv(self.csv_file, header=None, names=col_names)
        self.data_dir = data_dir
        self.labels_map = labels_map

    def determine_max_num_columns(self):
        with open(self.csv_file) as f:
            lines = f.readlines()
            max_cols = 0
            for line in lines:
                length = len(line.split(','))
                max_cols = length if length > max_cols else max_cols
        return max_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        num_segments = self.df.iloc[idx].count() - 2
        segment_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        segments = np.zeros((num_segments, self.num_frames_per_segment, 320, 320, 3))  # This is cheating
        for root, dirs, files in os.walk(segment_path):
            for i, file in enumerate(files):
                segment = os.path.abspath(os.path.join(root, file))
                cap = cv2.VideoCapture(segment)
                frames = np.zeros((self.num_frames_per_segment, 320, 320, 3))
                for j in range(self.num_frames_per_segment):
                    ret, frame = cap.read()
                    if ret:
                        frames[j] = frame
                segments[i] = frames

        label = self.df.iloc[idx, num_segments + 1]
        label_index = self.labels_map[label]
        label_tensor = torch.zeros((len(self.labels_map),))
        label_tensor[label_index] = 1

        return label_tensor, torch.tensor(segments, dtype=torch.float32)


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


class WLASLDataset(Dataset):
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
        frames = frames.reshape((frames.shape[3], frames.shape[0], frames.shape[1], frames.shape[2])) / 255.

        label = self.df.iloc[idx, 1]
        label_index = self.labels_map[label]
        label_tensor = torch.zeros((len(self.labels_map),))
        label_tensor[label_index] = 1

        return label_tensor, frames
