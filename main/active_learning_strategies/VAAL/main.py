import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
import mnist_net
from solver import Solver
from utils import *
import arguments

import time

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
        ])

def mnist_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

def main(args):
    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)

        args.num_images = 50000
        args.num_val = 5000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_val = 5000
        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
        
    elif args.dataset == 'mnist':
        test_dataloader = data.DataLoader(
                datasets.MNIST(args.data_path, download=True, transform=mnist_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)
        
        train_dataset = MNIST(args.data_path)
        
        args.num_val = 6000
        args.num_images = 60000
        args.budget = 10
        args.initial_budget = 50
        args.num_classes = 10
    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))
    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)
            
    args.cuda = True
    solver = Solver(args, test_dataloader)

    if args.dataset != 'mnist':
        splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    else:
        splits = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    current_indices = list(initial_indices)

    accuracies = []
    train_times = []
    sel_times = []
    
    
    for split in splits:
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        if args.dataset == 'mnist':
            task_model = mnist_net.MnistNet()
            print(args.latent_dim)
            vae = model.VAE(args.latent_dim, nc=1)
        else:
            task_model = vgg.vgg16_bn(num_classes=args.num_classes)
            vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim)

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        # train the models on the current data
        start_train = time.time()
        acc, vae, discriminator = solver.train(querry_dataloader,
                                               val_dataloader,
                                               task_model, 
                                               vae, 
                                               discriminator,
                                               unlabeled_dataloader)
        end_train = time.time()
        train_time = end_train - start_train
        print("Train time:", train_time)
        train_times.append(train_time)
        
        print('Final accuracy with {} samples is: {:.2f}'.format(int(split), acc))
        accuracies.append(acc)

        start_sampling = time.time()
        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        end_sampling = time.time()
        sampling_time = end_sampling - start_sampling
        sel_times.append(sampling_time)
        
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

    print("FINAL ACCS:", accuracies)
    print("FINAL TRAIN TIMES:", train_times)
    print("FINAL SEL. TIMES", sel_times)
    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)