import os
import argparse
import numpy as np
import random
import math
import numbers
import cv2
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler


class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):

        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i + h, j:j + w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i + th, j:j + tw, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames_from_video(vid_root, vid, start, num):
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

        label = np.zeros((num_classes, num_frames), np.float32)

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


class WLASLDataset(torch.utils.data.Dataset):

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

        # Sometimes videos load incorrectly. This is a workaround for that
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


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=(7, 7, 7),
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=(1, 1, 1), padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=(3, 3, 3), padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=(1, 1, 1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x, pretrained=False, n_tune_layers=-1):
        if pretrained:
            assert n_tune_layers >= 0

            freeze_endpoints = self.VALID_ENDPOINTS[:-n_tune_layers]
            tune_endpoints = self.VALID_ENDPOINTS[-n_tune_layers:]
        else:
            freeze_endpoints = []
            tune_endpoints = self.VALID_ENDPOINTS

        # backbone, no gradient part
        with torch.no_grad():
            for end_point in freeze_endpoints:
                if end_point in self.end_points:
                    x = self._modules[end_point](x)

        # backbone, gradient part
        for end_point in tune_endpoints:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        # head
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(root, train_split, weights):
    num_epochs_default = 400
    max_steps_default = 64000
    steps_per_update_default = 1
    batch_size_default = 6
    learning_rate_default = 0.0001
    random_seed_default = 0
    model_dir_default = "checkpoints"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.", default=0)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--max_steps", type=int, help="Max steps per epoch.",
                        default=max_steps_default)
    parser.add_argument("--steps_per_update", type=int, help="Number of steps to accumulate gradient before backprop.",
                        default=steps_per_update_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    argv = parser.parse_args()

    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    max_steps = argv.max_steps
    steps_per_update = argv.steps_per_update
    batch_size = argv.batch_size // WORLD_SIZE
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir

    print('Args:')
    print(f'LOCAL_RANK: {LOCAL_RANK}')
    print(f'WORLD_SIZE: {WORLD_SIZE}')
    print(f'WORLD_RANK: {WORLD_RANK}')
    print(f'num_epochs: {num_epochs}')
    print(f'batch_size: {batch_size}')
    print(f'learning_rate: {learning_rate}')
    print(f'random_seed: {random_seed}')
    print(f'model_dir: {model_dir}')

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    print("Init process group")
    torch.distributed.init_process_group(backend="gloo", rank=WORLD_RANK, world_size=WORLD_SIZE)
    print("Process group initialized")

    # setup dataset
    train_transforms = transforms.Compose([RandomCrop(224)])
    test_transforms = transforms.Compose([CenterCrop(224)])

    print("Loading train dataset")
    dataset = WLASLDataset(train_split, 'train', root, train_transforms)
    print("Loaded train dataset")
    train_sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    print("Loading val dataset")
    val_dataset = WLASLDataset(train_split, 'test', root, test_transforms)
    print("Loaded val dataset")
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}

    # device
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    print(f'Using GPU: {device}')

    # setup the model
    num_classes = dataset.num_classes
    i3d = InceptionI3d(num_classes, in_channels=3)

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    i3d.to(device)
    ddp_i3d = torch.nn.parallel.DistributedDataParallel(i3d, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    print("Created distributed model")

    lr = learning_rate
    weight_decay = 1e-8
    optimizer = optim.Adam(ddp_i3d.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps = 0
    epoch = 0

    best_val_score = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    print("Begining to train and validate")
    while steps < max_steps and epoch < num_epochs:
        print(f'Step {steps}/{max_steps}')
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                ddp_i3d.train(True)
            else:
                ddp_i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1:
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data

                if vid == -9999:
                    continue

                # wrap them in Variable
                inputs = inputs.to(device)
                t = inputs.size(2)
                labels = labels.to(device)

                per_frame_logits = ddp_i3d(inputs, pretrained=False)
                # upsample to input size
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(predictions, gt)
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / steps_per_update
                tot_loss += loss.data.item()
                loss.backward()

                if phase == 'train':
                    steps += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print(
                            f'Epoch {epoch} {phase} Loc Loss: {tot_loc_loss / (10 * steps_per_update):.4f} Cls Loss: {tot_cls_loss / (10 * steps_per_update):.4f} Tot Loss: {tot_loss / 10:.4f} Accu :{acc:.4f} '
                        )
                        num_iter = 0
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if (val_score > best_val_score or epoch % 2 == 0) and WORLD_RANK == 0:
                    best_val_score = val_score
                    model_name = os.path.join(model_dir, "nslt_" + str(num_classes) + "_" + str(steps).zfill(
                        6) + '_%3f.pt' % val_score)

                    torch.save(ddp_i3d.module.state_dict(), model_name)
                    print(model_name)

                    print(
                        f'VALIDATION: {phase} Loc Loss: {tot_loc_loss / num_iter:.4f} Cls Loss: {tot_cls_loss / num_iter:.4f} Tot Loss: {(tot_loss * steps_per_update) / num_iter:.4f} Accu :{val_score:.4f}')
                scheduler.step(tot_loss * steps_per_update / num_iter)


if __name__ == '__main__':
    root = '../data'
    train_split = 'train_test_val.json'
    print(root, train_split)
    main(root=root, train_split=train_split, weights='checkpoints/nslt_2000_044946_0.142461.pt')
