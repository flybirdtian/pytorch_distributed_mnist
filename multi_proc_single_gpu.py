from __future__ import division, print_function

import argparse
import random
import time
import warnings
import os
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
import torch.multiprocessing as mp

best_acc = 0

def distributed_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    @torch.no_grad()
    def update(self, output, target):
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        for data, target in self.train_loader:
            data = data.cuda(self.device)
            target = target.cuda(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)

        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        for data, target in self.test_loader:
            data = data.cuda(self.device)
            target = target.cuda(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            test_loss.update(loss.item(), data.size(0))
            test_acc.update(output, target)

        return test_loss, test_acc


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class MNISTDataLoader(data.DataLoader):

    def __init__(self, root, batch_size, num_workers, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.MNIST(
            root, train=train, transform=transform, download=True)

        self.train = train

        self.sampler = None
        if train and distributed_is_initialized():
            self.sampler = data.DistributedSampler(dataset)

        if train:
            shuffle = (self.sampler is None)
        else:
            shuffle = False

        # Note: for small dataset, it's better to set pin_memory=False, otherwise for large dataset, set pin_memory=true
        super(MNISTDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self.sampler, num_workers=num_workers, pin_memory=False
        )

    def set_sample_epoch(self, epoch=0):
        if self.train and self.sampler is not None:
            self.sampler.set_epoch(epoch)

def run(args):
    global best_acc

    # initialize the process group
    dist.init_process_group(
        backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    ngpus = torch.cuda.device_count()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus)
    args.workers = int((args.workers + ngpus - 1) / ngpus)

    # rank = dist.get_rank()
    rank = args.rank

    gpu_id = rank
    device = torch.device("cuda", gpu_id)

    print("rank: {}, device count: {}, workers:{}".format(rank, ngpus, args.workers))

    model = Net()
    if distributed_is_initialized():
        model.cuda(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu_id])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            # Map model to be loaded to current gpu.
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch']

            best_acc = checkpoint['best_acc']
            print("best_acc: {}".format(best_acc))

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader = MNISTDataLoader(
        args.root, args.batch_size, num_workers=args.workers, train=True)
    test_loader = MNISTDataLoader(
        args.root, args.batch_size, num_workers=args.workers, train=False)

    trainer = Trainer(model, optimizer, train_loader, test_loader, device)

    if args.evaluate:
        test_loss, test_acc = trainer.evaluate()
        print('test loss: {}, test acc: {}.'.format(test_loss, test_acc))
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.set_sample_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, train_acc = trainer.train()
        test_loss, test_acc = trainer.evaluate()

        print(
            'Epoch: {}/{},'.format(epoch, args.epochs),
            'train loss: {}, train acc: {},'.format(train_loss, train_acc),
            'test loss: {}, test acc: {}.'.format(test_loss, test_acc)
        )

        # remember best acc and save checkpoint
        is_best = test_acc.accuracy > best_acc
        best_acc = max(test_acc.accuracy, best_acc)

        # only save checkpoints in the first gpu
        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, epoch):
    chk_dir = 'checkpoints'
    if not os.path.exists(chk_dir):
        os.makedirs(chk_dir)
    filename=os.path.join(chk_dir, 'checkpoint_{}.pth.tar'.format(epoch))
    torch.save(state, filename)
    if is_best:
        bestfilename = os.path.join(chk_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, bestfilename)

def run_spawn(proc_id, args):
    # for single node with multiple gpus, if using torch.multiprocessing.spawn, rank= process_id
    args.rank = proc_id
    run(args)

def run_dist_launch(args):
     # for single node with multiple gpus, if using torch.distributed.launch, rank= local_rank
    args.rank = args.local_rank
    run(args)


def demo_spawn(ngpus, args):
    mp.spawn(run_spawn, args=(args,), nprocs=ngpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='mini-batch size(default: 256), this is the '
                        'total batch size of all GPUs on the current node '
                        'when use Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--backend', type=str, default='nccl',
                        help='Name of the backend to use.')

    # when use torch.distributed.launch, you need to include 'local_rank' arguments,
    # otherwise it will cause an error
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('-i',
                        '--init-method',
                        type=str,
                        default='tcp://127.0.0.1:23456',
                        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=1,
                        help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='Rank of the current process.')

    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus = torch.cuda.device_count()
    assert(args.world_size == ngpus)

    # way 1: using torch.distributed.launch, then in a terminal, run:
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 multi_proc_single_gpu.py --world-size 4 --workers 4
    # run_dist_launch(args)

    # way 2: using torch.multiprocessing.spawn, then in a terminal, run:
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_proc_single_gpu.py --world-size 4 --workers 4
    demo_spawn(ngpus, args)
