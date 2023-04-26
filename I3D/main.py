import os
import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

from I3D import InceptionI3d
import videotransforms
from nslt_dataset import NSLT as Dataset


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(root, train_split, weights):
    num_epochs_default = 400
    max_steps_default = 64000
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
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.", default=False)
    argv = parser.parse_args()

    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    max_steps = argv.max_steps
    batch_size = argv.batch_size // WORLD_SIZE
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    resume = argv.resume

    print('Args:')
    print(f'LOCAL_RANK: {LOCAL_RANK}')
    print(f'WORLD_SIZE: {WORLD_SIZE}')
    print(f'WORLD_RANK: {WORLD_RANK}')
    print(f'num_epochs: {num_epochs}')
    print(f'batch_size: {batch_size}')
    print(f'learning_rate: {learning_rate}')
    print(f'random_seed: {random_seed}')
    print(f'model_dir: {model_dir}')
    print(f'resume: {resume}')

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    print("Init process group")
    torch.distributed.init_process_group(backend="gloo", rank=WORLD_RANK, world_size=WORLD_SIZE)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224)])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, train_transforms)
    train_sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                             shuffle=True)

    val_dataset = Dataset(train_split, 'test', root, test_transforms)
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                                 shuffle=True, num_workers=2, pin_memory=False)

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

    lr = learning_rate
    weight_decay = 1e-8
    optimizer = optim.Adam(ddp_i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = 1  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < max_steps and epoch < num_epochs:
        print('Step {}/{}'.format(steps, max_steps))
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

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int16)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1:
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = inputs.to(device)
                t = inputs.size(2)
                labels = labels.to(device)

                per_frame_logits = ddp_i3d(inputs, pretrained=False)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print(
                            f'Epoch {epoch} Iter {num_iter} {phase} Loc Loss: {tot_loc_loss / (10 * num_steps_per_update):.4f} Cls Loss: {tot_cls_loss / (10 * num_steps_per_update):.4f} Tot Loss: {tot_loss / 10:.4f} Accu :{acc:.4f} '
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
                        f'VALIDATION: {phase} Loc Loss: {tot_loc_loss / num_iter:.4f} Cls Loss: {tot_cls_loss / num_iter:.4f} Tot Loss: {(tot_loss * num_steps_per_update) / num_iter:.4f} Accu :{val_score:.4f}')
                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == '__main__':
    root = '../data'
    train_split = 'preprocess/nslt_2000.json'
    print(root, train_split)
    main(root=root, train_split=train_split, weights=None)
