from numpy import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
# from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from network import PredictNetwork
import torchvision.transforms as transforms
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import config

class IVAL_Dataset(Dataset):
    def __init__(self):
        self.file_path = '.\\dataset'
        self.image_list = os.listdir(self.file_path)
        # self.transforms = transforms.Compose([
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        #                                     ])

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        data_path = os.path.join(self.file_path, self.image_list[index])
        f_read = open(data_path, 'rb')
        load_pkl = pickle.load(f_read)
        f_read.close()
        image_tensor = self.transforms(load_pkl['image'])
        vel = load_pkl['vel']
        action_series = load_pkl['action_series']
        label = load_pkl['label']
        
        return image_tensor, vel, action_series, label

    def __len__(self):
        return len(self.image_list)

def cal_acc(mean_loss_list):
    if len(mean_loss_list)<10:
        return mean(mean_loss_list)
    else:
        return mean(mean_loss_list[-10:])

def plot(mean_loss_list, mean_loss_acc_list):
    plt.figure(figsize=(12, 8))
    plt.plot(mean_loss_list)
    plt.plot(mean_loss_acc_list)
    plt.xlabel('epoch')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('loss')
    plt.show()
    plt.close()


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# use GPU
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
# print('device:', device)

model = PredictNetwork().to(device)
optimizer  = optim.Adam(model.parameters(), lr=config.pred_net_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
criterion = nn.MSELoss()

ival_dataset = IVAL_Dataset()  
IVAL_dataloader = DataLoader(dataset=ival_dataset, batch_size=config.pred_net_batch_size, shuffle=True, num_workers=2, drop_last=True)

def main():
    max_epochs = config.pred_max_epochs
    max_steps = len(IVAL_dataloader)
    min_loss = np.float64('inf')
    mean_loss_list = []
    mean_loss_acc_list = []

    print('Start training...')
    model.train()
    for epoch in range(max_epochs):
        epoch_loss = []
        print('\n******** epoch [%s / %s] ********' % (epoch, max_epochs))
        for step,(batch_image_tensor, batch_vel, batch_action_series, batch_label) in enumerate(IVAL_dataloader):
            batch_image_tensor = batch_image_tensor.float().to(device)
            batch_vel = batch_vel.float().to(device)
            batch_action_series = batch_action_series.float().to(device)
            batch_label = batch_label.float().to(device)
            
            optimizer.zero_grad()
            output = model(batch_image_tensor, batch_vel, batch_action_series)

            loss = criterion(output,batch_label)
            epoch_loss.append(float(loss.cpu()))
            loss.backward()

            optimizer.step()

            if step>0 and step % 200 == 0:
                print('%d%% [%s / %s], loss: %.6f' % (math.ceil(step/max_steps*100), step, max_steps, loss))
        
        scheduler.step()
        mean_loss = mean(epoch_loss)
        mean_loss_list.append(mean_loss)
        mean_loss_acc_list.append(cal_acc(mean_loss_list))
        print('epoch %s average loss: %.6f' % (epoch, mean_loss))
        if mean_loss < min_loss:
            min_loss = mean_loss
            ckpt_path = os.path.join('.\\checkpoints', 'ep-%d.pth' % epoch)
            torch.save({
                'model': model
            }, ckpt_path)
            print('------ Model of Epoch %d has been saved ------' % epoch)
    print('Finish training.')
    plot(mean_loss_list, mean_loss_acc_list)
    return

if __name__ == '__main__':
    main()
