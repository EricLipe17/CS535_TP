import math
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

cfm = np.load('confusion_matrix.npy')
a = np.argmax(cfm, axis=1)

for i in range(a.shape[0]):
    print(i, a[i])

# with open('../progress.txt') as f:
#     loss = list()
#     accuracy = list()
#     for line in f.readlines():
#         split = line.split()
#         if len(split) != 14:
#             continue
#         loss.append(float(split[11]))
#         accuracy.append(float(split[13].strip(':')))
#
# sns.set_style("darkgrid")
# fig, axs = plt.subplots(2)
# fig.suptitle('Training Loss & Accuracy')
# axs[0].plot(loss)
# axs[1].plot(accuracy)
#
# start = max(loss)
# stop = math.floor(min(loss))
# loss_yticks = [i for i in np.round(np.arange(start, stop, -0.001), 3)]
# start = int(math.floor(min(accuracy)))
# stop = int(math.ceil(max(accuracy))) + 0.1
# acc_yticks = [i for i in np.round(np.arange(start, stop, 0.1), 3)]
#
# axs[0].set(xlabel='# of Steps', ylabel='Loss')
# axs[0].set_yticks(loss_yticks)
# axs[0].set_yticklabels(loss_yticks, fontsize=12)
# axs[1].set(xlabel='# of Steps', ylabel='Accuracy')
# axs[1].set_yticks(acc_yticks)
# axs[1].set_yticklabels(acc_yticks, fontsize=12)
# fig.tight_layout(pad=1)
# fig.savefig('train_accuracy_loss.png')
