from __future__ import print_function

import os
import six
import math
import torch
import argparse
import datetime
import callbacks
import torchvision

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from binary import BinaryNet, BigBinaryNet


#Command Line Args
parser = argparse.ArgumentParser(description='Binary Network')

parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TB',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--use_scheduler', type=bool, default=False, metavar='SCH',
                    help='Decay Learning Rate on Plateau (Default False)')

parser.add_argument('--optimizer', type=str, default='sgd', metavar='O',
                    help='Optimizer options are sgd, adam, rms_prop')

parser.add_argument('--momentum', type=float, default=0.5, metavar='MO',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log_interval', type=int, default=100, metavar='I',
                    help="""how many batches to wait before logging detailed
                            training status, 0 means never log """)

parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='Options are mnist and fashion_mnist')

parser.add_argument('--data_dir', type=str, default='../data/', metavar='F',
                    help='Where to put data')

parser.add_argument('--name', type=str, default='', metavar='N',
                    help="""A name for this training run, this
                            affects the directory so use underscores and not spaces.""")

parser.add_argument('--print_log', action='store_true', default=False,
                    help='prints the csv log when training is complete')

parser.add_argument('--writemodel', type=bool, default=False, metavar='TDSK',
                    help='Save model to disk?')

parser.add_argument('--model', type=str, default='default', metavar='M')

parser.add_argument('--clip', type=bool, default=False, metavar='CLP',
                    help='Clip model weights to -1, 1 after each iteration (default: False)')

parser.add_argument('--dropout', type=bool, default=False, metavar='DPT')

parser.add_argument('--gradclip', type=bool, default=False, metavar='CGD',
                    help='Cancel gradients in BLU when magnitude is > 1 (default: False)')

parser.add_argument('--transform', type=bool, default=False, metavar='TFM',
                    help='Use data augmentation transforms (default: False)')

required = object()


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Add a timestamp to your training run's name.
    """
    # http://stackoverflow.com/a/5215012/99379
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def prepareDatasetAndLogging(args):

    # choose the dataset
    if args.dataset == 'mnist':
        DatasetClass = datasets.MNIST
    elif args.dataset == 'fashion_mnist':
        DatasetClass = datasets.FashionMNIST
    else:
        raise ValueError('unknown dataset: ' + args.dataset + ' try mnist or fashion_mnist')

    training_run_name = timeStamped(args.dataset + '_' + args.name)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Create the dataset, mnist or fasion_mnist
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    training_run_dir = os.path.join(args.data_dir, training_run_name)

    if args.transform:
        print('Using Data Augmentation!')
        train_dataset = DatasetClass(dataset_dir,
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                        transforms.RandomRotation(5),
                                        transforms.ColorJitter(),
                                        transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
                                    )
    else:
        train_dataset = DatasetClass(dataset_dir,
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
                                    )

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_dataset = DatasetClass(dataset_dir,
                                train=False,
                                transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,),
                                                                 (0.3081,))
                                            ])
                                )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              **kwargs)

    # Set up visualization and progress status update code
    callback_params = {'epochs': args.epochs,
                       'samples': len(train_loader) * args.batch_size,
                       'steps': len(train_loader),
                       'metrics': {'acc': np.array([]),
                                   'loss': np.array([]),
                                   'val_acc': np.array([]),
                                   'val_loss': np.array([])}
                      }

    if args.print_log:
        output_on_train_end = os.sys.stdout
    else:
        output_on_train_end = None

    callbacklist = callbacks.CallbackList([callbacks.BaseLogger(),
                                           callbacks.TQDMCallback(),
                                           callbacks.CSVLogger(filename=training_run_dir + training_run_name + '.csv',
                                           output_on_train_end=output_on_train_end)])

    callbacklist.set_params(callback_params)

    tensorboard_writer = SummaryWriter(log_dir=training_run_dir,
                                       comment=args.dataset + '_embedding_training')

    return tensorboard_writer, callbacklist, train_loader, test_loader


def chooseModel(args, cuda=True):
    model_name = args.model
    if model_name == 'default':
        model = BinaryNet(use_dropout=args.dropout,
                          use_grad_clip=args.gradclip)
    elif model_name == 'big':
        model = BigBinaryNet(use_dropout=args.dropout,
                              use_grad_clip=args.gradclip)
    else:
        raise ValueError('Unknown model type: ' + model_name)
    if args.cuda:
        model.cuda()
    return model


def chooseOptimizer(model, optimizer='sgd', args=None):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr)
    else:
        raise ValueError('Unsupported optimizer: ' + args.optimizer)
    return optimizer


def train(model,
          optimizer,
          train_loader,
          tensorboard_writer,
          callbacklist,
          epoch,
          total_minibatch_count,
          args):

    # Training
    model.train()
    correct_count = np.array(0)
    for batch_idx, (data, target) in enumerate(train_loader):

        callbacklist.on_batch_begin(batch_idx)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.view(-1, 784)), Variable(target)
        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        loss = F.cross_entropy(output, target)

        # Backpropagation step
        loss.backward()
        if args.clip:
            model.clip()
        optimizer.step()

        # The batch has ended, determine the
        # accuracy of the predicted outputs
        _, argmax = torch.max(output, 1)

        # target labels and predictions are
        # categorical values from 0 to 9.
        accuracy = (target == argmax.squeeze()).float().mean()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_count += pred.eq(target.data.view_as(pred)).cpu().sum()

        batch_logs = {
            'loss': np.array(loss.data[0]),
            'acc': np.array(accuracy.data[0]),
            'size': np.array(len(target))
        }

        batch_logs['batch'] = np.array(batch_idx)
        callbacklist.on_batch_end(batch_idx, batch_logs)

        if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:
            # put all the logs in tensorboard
            for name, value in six.iteritems(batch_logs):
                tensorboard_writer.add_scalar(name,
                                              value,
                                              global_step=total_minibatch_count)

            # put all the parameters in tensorboard histograms
            for name, param in model.named_parameters():
                name = name.replace('.', '/')

                tensorboard_writer.add_histogram(name,
                                                 param.data.cpu().numpy(),
                                                 global_step=total_minibatch_count)

                tensorboard_writer.add_histogram(name + '/gradient',
                                                 param.grad.data.cpu().numpy(),
                                                 global_step=total_minibatch_count)

        total_minibatch_count += 1

    return total_minibatch_count


def test(model,
         test_loader,
         tensorboard_writer,
         callbacklist,
         epoch,
         total_minibatch_count):

    # Validation Testing
    model.eval()
    test_loss = 0
    correct = 0
    progress_bar = tqdm(test_loader, desc='Validation')
    for data, target in progress_bar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.view(-1, 784), volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_size = np.array(len(test_loader.dataset), np.float32)
    test_loss /= test_size

    acc = np.array(correct, np.float32) / test_size
    epoch_logs = {'val_loss': np.array(test_loss),
                  'val_acc': np.array(acc)}

    for name, value in six.iteritems(epoch_logs):
        tensorboard_writer.add_scalar(name,
                                      value,
                                      global_step=total_minibatch_count)

    callbacklist.on_epoch_end(epoch, epoch_logs)
    progress_bar.write(
        'Epoch: {} - validation test results - Average val_loss: {:.4f}, val_acc: {}/{} ({:.2f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc


def run_experiment(args):
    total_minibatch_count = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    epochs_to_run = args.epochs
    tensorboard_writer, callbacklist, train_loader, test_loader = prepareDatasetAndLogging(args)

    optimizer = None
    model = chooseModel(args)

    optimizer = chooseOptimizer(model,
                                args.optimizer,
                                args)

    # Run the primary training loop, starting with validation accuracy of 0
    val_acc = 0
    callbacklist.on_train_begin()
    for epoch in range(1, epochs_to_run + 1):
        callbacklist.on_epoch_begin(epoch)

        # train for 1 epoch
        total_minibatch_count = train(model,
                                      optimizer,
                                      train_loader,
                                      tensorboard_writer,
                                      callbacklist,
                                      epoch,
                                      total_minibatch_count,
                                      args)

        # validate progress on test dataset
        val_acc = test(model,
                       test_loader,
                       tensorboard_writer,
                       callbacklist,
                       epoch,
                       total_minibatch_count)

    callbacklist.on_train_end()
    tensorboard_writer.close()

    if args.writemodel:
        torch.save(model.state_dict(), './current.model')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    run_experiment(args)
