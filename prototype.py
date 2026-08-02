import copy
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from my_dataset_class import Mydataset
from model import RES_GAT, FocalLoss
import util


# Argument parsing (GAT model params, kept for compatibility)
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
args, _ = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def train_model(epochs, batch_size, use_cuda, dset_folder, disable_tqdm=False):
    print("Loading dataset...")
    dset = Mydataset('./data', train=True, transform=util.get_graph_from_image)

    valid_split = 0.2
    valid_len = int(len(dset) * valid_split)
    train_len = len(dset) - valid_len
    dset_train, dset_valid = torch.utils.data.random_split(dset, [train_len, valid_len])

    dset_train_loader, dset_valid_loader = map(
        lambda ds: torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=True, collate_fn=util.batch_graphs,
        ),
        [dset_train, dset_valid]
    )

    model = RES_GAT(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    if use_cuda:
        model = model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    best_valid_acc = 0.
    best_model = copy.deepcopy(model)

    for e in tqdm(range(epochs), total=epochs, desc="Epoch ", disable=disable_tqdm):
        train_losses, train_accs, train_acc1s, train_pres, train_recalls, train_f1s = \
            util.train(model, opt, dset_train_loader, use_cuda=use_cuda, disable_tqdm=disable_tqdm)

        last_epoch_train_loss = np.mean(train_losses)
        last_epoch_train_acc = 100 * np.mean(train_accs)
        last_epoch_train_acc1 = 100 * np.mean(train_acc1s)
        last_epoch_train_pre = 100 * np.mean(train_pres)
        last_epoch_train_recall = 100 * np.mean(train_recalls)
        last_epoch_train_f1 = 100 * np.mean(train_f1s)

        valid_accs, valid_acc1s, valid_pres, valid_recalls, valid_f1s = \
            util.test(model, dset_valid_loader, use_cuda, desc="Validation ", disable_tqdm=disable_tqdm)

        last_epoch_valid_acc = 100 * np.mean(valid_accs)
        last_epoch_valid_acc1 = 100 * np.mean(valid_acc1s)
        last_epoch_valid_pre = 100 * np.mean(valid_pres)
        last_epoch_valid_recall = 100 * np.mean(valid_recalls)
        last_epoch_valid_f1 = 100 * np.mean(valid_f1s)

        if last_epoch_valid_acc > best_valid_acc:
            best_valid_acc = last_epoch_valid_acc
            best_model = copy.deepcopy(model)

        log = ("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {t_acc1:.2f}% {t_pre:.2f}% "
               "{t_recall:.2f}% {t_f1:.2f}% {v_acc:.2f}% {v_acc1:.2f}% "
               "{v_pre:.2f}% {v_recall:.2f}% {v_f1:.2f}%").format(
            loss=last_epoch_train_loss,
            t_acc=last_epoch_train_acc, t_acc1=last_epoch_train_acc1,
            t_pre=last_epoch_train_pre, t_recall=last_epoch_train_recall, t_f1=last_epoch_train_f1,
            v_acc=last_epoch_valid_acc, v_acc1=last_epoch_valid_acc1,
            v_pre=last_epoch_valid_pre, v_recall=last_epoch_valid_recall, v_f1=last_epoch_valid_f1,
        )
        tqdm.write(log)

    util.save_model("best", best_model)
    print(f"Training complete. Best validation accuracy: {best_valid_acc:.2f}%")


def main(train=True, test=False, epochs=500, batch_size=32, use_cuda=True, disable_tqdm=False, dset_folder="./cifar10"):
    use_cuda = use_cuda and torch.cuda.is_available()
    if train:
        train_model(epochs=epochs, batch_size=batch_size, use_cuda=use_cuda, dset_folder=dset_folder, disable_tqdm=disable_tqdm)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
