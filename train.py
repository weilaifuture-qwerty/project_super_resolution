import torch
import torch.nn as nn
from vgg import vgg16
from network import ImageTransformNet
from loss import get_loss, gram
import os

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils as tvutils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import logging
import time
from io import BytesIO
from tqdm import tqdm


BATCH_SIZE = 4
NUM_ITERATION = 40000
LEARNING_RATE = 1e-3
NUM_EPOCHES = 2
LOG_EVERY = 200
SAMPLES_EVERY = 1000
STEP_LR = 2e-3

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers): 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>8s} [{global_step:07d}]: {value:5f}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def transform_back(data, save_dir):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.detach().clone().cpu().numpy()[0]
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(save_dir)

def train_model():
    device = torch.device('mps')
    # setup_logging()
    torch.set_num_threads(4)
    writer = SummaryWriter("./log.txt", max_queue=1000, flush_secs=120)
    model = ImageTransformNet()
    model.train()

    transform = transforms.Compose([
        transforms.Resize(36),           
        transforms.CenterCrop(36),      
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder("coco", transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    vgg = vgg16()

    transform_style = transforms.Compose([  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(512),           
        transforms.CenterCrop(512),      
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    clock = Image.open("./test_image/clock.jpg")
    clock = transform_test(clock)
    clock = clock.to(device)
    clock = clock.repeat(1, 1, 1, 1)


    cake = Image.open("./test_image/cake.jpg")
    cake = transform_test(cake)
    cake = cake.to(device)
    cake = cake.repeat(1, 1, 1, 1)

    style_image = Image.open("./style_image/EVA.jpg")
    style_image = transform_style(style_image)
    style_image = style_image.to(device)
    style_features = vgg(style_image)


    iterations = 0
    train_loss = []
    for epoch in range(1, NUM_EPOCHES + 1):
        batch_idx = 0
        for data, _ in tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch"):
            batch_idx = batch_idx + 1
            data = data.to(device)
            optimizer.zero_grad()

            y_hat = model(data)

            y_hat_features = vgg(y_hat)
            x_features = vgg(data)

            tv_loss = torch.sum(torch.abs(y_hat[:, :, :, :-1] - y_hat[:, :, :, 1:])) + torch.sum(torch.abs(y_hat[:, :, :-1, :] - y_hat[:, :, 1:, :]))

            loss, style_loss, content_loss = get_loss(y_hat_features, style_features, x_features)
            loss += 1e-7 * tv_loss
            loss.backward()
            optimizer.step()

            train_loss += [loss.item()]
            iterations += 1
            # print(iterations)

            if iterations % LOG_EVERY == 0:
                writer.add_scalar('loss', np.mean(train_loss), iterations)
                print('loss', np.mean(train_loss), iterations, tv_loss.item(), style_loss.item(), content_loss.item(), loss)
                train_loss = []
            
            if iterations % SAMPLES_EVERY == 0:
                model.eval()
                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                
                output_clock = model(clock).cpu()
                clock_path = "visualization/eva/clock/clock_%d_%05d.jpg" %(epoch, batch_idx)
                transform_back(output_clock, clock_path)

                output_cake = model(cake).cpu()
                cake_path = "visualization/eva/cake/cake_%d_%05d.jpg" %(epoch, batch_idx)
                transform_back(output_cake, cake_path)
            
            if iterations >= 200000:
                break
    
    torch.save(model, './model')
                