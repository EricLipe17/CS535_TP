import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import videotransforms
import numpy as np
from I3D import InceptionI3d
from nslt_dataset import NSLT as Dataset


def main(root, train_split, weights):
    num_epochs_default = 400
    max_steps_default = 64000
    steps_per_update_default = 1
    batch_size_default = 6
    learning_rate_default = 0.0001
    random_seed_default = 0
    model_dir_default = "checkpoints"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

    num_epochs = argv.num_epochs
    max_steps = argv.max_steps
    steps_per_update = argv.steps_per_update
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224)])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = Dataset(train_split, 'test', root, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}

    # setup the model
    i3d = InceptionI3d(2000, in_channels=3)

    num_classes = dataset.num_classes

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using GPU: {device}')

    i3d.to(device)
    i3d = nn.DataParallel(i3d)

    weight_decay = 1e-8
    optimizer = optim.Adam(i3d.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps = 0
    epoch = 0

    best_val_score = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < max_steps and epoch < num_epochs:
        print(f'Step {steps}/{max_steps}')
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)

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

                # wrap them in Variable
                inputs = inputs.to(device)
                t = inputs.size(2)
                labels = labels.to(device)

                per_frame_logits = i3d(inputs, pretrained=False)
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

                if num_iter == steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print(
                            'Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                 phase,
                                                                                                                 tot_loc_loss / (10 * steps_per_update),
                                                                                                                 tot_cls_loss / (10 * steps_per_update),
                                                                                                                 tot_loss / 10,
                                                                                                                 acc))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = os.path.join(model_dir, "nslt_" + str(num_classes) + "_" + str(steps).zfill(
                        6) + '_%3f.pt' % val_score)

                    torch.save(i3d.module.state_dict(), model_name)
                    print(model_name)

                print('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                              tot_loc_loss / num_iter,
                                                                                                              tot_cls_loss / num_iter,
                                                                                                              (tot_loss * steps_per_update) / num_iter,
                                                                                                              val_score
                                                                                                              ))

                scheduler.step(tot_loss * steps_per_update / num_iter)


if __name__ == '__main__':
    root = '../data'
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_2000.json'

    main(root=root, train_split=train_split, weights='checkpoints/nslt_2000_002497_0.022161.pt')
