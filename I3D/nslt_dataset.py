import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch


def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
    video_path = os.path.abspath(os.path.join(vid_root, vid + '.mp4'))

    cap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = cap.read()

        if success:
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

            img = (img / 255.) * 2 - 1

            frames.append(img)

    if not frames:
        print(f"\nFrames length zero for video: {vid}\n")
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, num_classes):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    skipped_videos = 0
    for vid in data.keys():
        if split == 'train':
            if data[vid]['subset'] not in ['train', 'val']:
                continue
        else:
            if data[vid]['subset'] != 'test':
                continue

        vid_root = root

        video_path = os.path.abspath(os.path.join(vid_root, vid + '.mp4'))
        if not os.path.exists(video_path):
            continue

        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames < 9:
            print("Skip video ", vid)
            skipped_videos += 1
            continue

        label = np.zeros((num_classes, num_frames), np.int8)

        for i in range(num_frames):
            cls = data[vid]['action'][0]
            label[cls][i] = 1

        config_num_frames = data[vid]['action'][2] - data[vid]['action'][1]
        if len(vid) == 5:
            dataset.append((vid, label, 0, config_num_frames))
        elif len(vid) == 6:
            dataset.append((vid, label, data[vid]['action'][1], config_num_frames))

    print("Skipped videos: ", skipped_videos)
    print(len(dataset))
    return dataset


def get_num_class(split_file):
    classes = set()
    content = json.load(open(split_file))
    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)
    return len(classes)


class NSLT(torch.utils.data.Dataset):

    def __init__(self, split_file, split, root, transforms=None):
        self.num_classes = get_num_class(split_file)

        self.data = make_dataset(split_file, split, root, num_classes=self.num_classes)
        self.split_file = split_file
        self.transforms = transforms
        self.root = root

        self.prev_frames = None
        self.prev_label = None
        self.prev_vid = None

    def __getitem__(self, index):
        vid, label, start_frame, num_frames = self.data[index]
        total_frames = 64
        try:
            start_f = random.randint(0, num_frames - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame

        # Sometimes videos loaded incorrectly. This is a workaround for that
        frames = load_rgb_frames_from_video(self.root, vid, start_f, total_frames)
        if not frames.any() or frames.shape[-1] != 3:
            print("No frames, returning zeros array")
            if self.prev_vid is not None:
                return self.prev_frames, self.prev_label, self.prev_vid
            return torch.zeros((3, total_frames, 224, 224)), torch.zeros((2000, total_frames)), -9999

        frames, label = self.pad(frames, label, total_frames)

        frames = self.transforms(frames)

        label = torch.from_numpy(label)
        frames = video_to_tensor(frames)

        self.prev_frames = frames
        self.prev_label = label
        self.prev_vid = vid

        return frames, label, vid

    def __len__(self):
        return len(self.data)

    def pad(self, frames, label, total_frames):
        padded_frames = frames
        if frames.shape[0] < total_frames:
            num_padding = total_frames - frames.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = frames[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_frames = np.concatenate([frames, pad], axis=0)
                else:
                    pad_img = frames[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_frames = np.concatenate([frames, pad], axis=0)

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_frames, label

    @staticmethod
    def pad_wrap(frames, label, total_frames):
        padded_frames = frames
        if frames.shape[0] < total_frames:
            num_padding = total_frames - frames.shape[0]

            if num_padding:
                pad = frames[:min(num_padding, frames.shape[0])]
                k = num_padding // frames.shape[0]
                tail = num_padding % frames.shape[0]

                pad2 = frames[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_frames = np.concatenate([frames, pad1, pad2], axis=0)
                else:
                    padded_frames = np.concatenate([frames, pad2], axis=0)

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_frames, label
