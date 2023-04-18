import cv2
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset


class WLASLSegmentDataset(Dataset):
    def __init__(self, data_dir, csv_file):
        self.csv_file = csv_file
        self.num_frames_per_segment = int(self.csv_file.split('_')[0])
        col_names = [i for i in range(self.determine_max_num_columns())]
        self.df = pd.read_csv(self.csv_file, header=None, names=col_names)
        self.data_dir = data_dir

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

        return label, segments
