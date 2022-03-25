import torch
import os
# from SampleActions import sample_actions
from self_dataset.dataset import IVA_Dataset
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
import config
from tqdm import tqdm
import time
import numpy as np

def process_time_serie(ola_action_series, sample, H, beta):
    actions = []
    for h in range(H):
        if h == 0:
            action_h = beta * (ola_action_series[h] + sample[:,h])
        else:
            action_h = beta * (ola_action_series[h] + sample[:,h]) + (1.0 - beta) * actions[-1]
        actions.append(action_h)
    return np.vstack(actions).T

def sample_actions(ola_action_series, N):
    H = 10
    beta = 0.6
    
    ola_action_series = np.vstack((ola_action_series[1:],ola_action_series[-1]))
    sample_steer = process_time_serie(ola_action_series[:,0], np.random.normal(loc=0.0, scale=1.0, size=(N,H)), H, beta)
    sample_throttle = process_time_serie(ola_action_series[:,1], np.random.normal(loc=0.0, scale=1.5, size=(N,H)), H, beta)
    sample_brake = process_time_serie(ola_action_series[:,2], np.random.normal(loc=0.0, scale=0.2, size=(N,H)), H, beta)
    
    return torch.FloatTensor(np.dstack((np.clip(sample_steer,-1.0,1.0), np.clip(sample_throttle,0.0,1.0), np.clip(sample_brake,0.0,1.0))))

def angle_of_vector(v1):
    with torch.no_grad():
        length_prod = torch.sqrt(torch.pow(v1[:,:,0], 2) + torch.pow(v1[:,:,1], 2))
        cos = v1[:,:,0] / (length_prod + 1e-6)
    return torch.acos(cos)

def cal_optim_action_series(output, pos, goal, sample_actions_series):#4096*10*5
    pi = 3.14159
    gamma = 2.0
    a_pos, a_lane = 1.0, 2.0
    
    with torch.no_grad():
        R_coll = output[:,:,1]
        
        v_pos = goal-output[:,:,2:4]
        ang_v = pos[2]/180.0*pi
        pos_angel = angle_of_vector(v_pos)
        pos_angel[v_pos[:,:,1]>0] *= -1.0
        angel_diff = torch.abs(ang_v - pos_angel)/pi
        R_pos = (1.0 - output[:,:,1]).mul(angel_diff) + output[:,:,1]
        
        R_lane = (1.0 - output[:,:,1]).mul(output[:,:,0]) + output[:,:,1]
        
        R = torch.sum((R_coll + a_pos* R_pos + a_lane * R_lane), dim=1)
        
        exp_gamma_R = torch.exp(R * gamma)
        
        optim_action_series = torch.sum(exp_gamma_R.repeat(3,10,1).transpose(0,2).mul(sample_actions_series), dim=0)/torch.sum(exp_gamma_R, dim=0)
    
    return optim_action_series

class plan(object):
    def __init__(self, model_name, N):
        print('initializing planner...')
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.N = N
        
        abs_path = os.path.abspath(r'.\\checkpoints')
        model_path = os.path.join(abs_path, model_name)
        self.model = torch.load(model_path)
        # self.model = torch.load('C:\\Users\\15662\\Desktop\\Silence\\2022Spring\\BADGR\\checkpoints\\ep-95.pth')
        self.model = self.model['model']
        self.model.to(self.device)
        print('done.')
        
    def update_actions(self, image, vel, pos, old_actions, goal):
        sample_actions_series = sample_actions(old_actions, self.N)
        iva_dataset = IVA_Dataset(image, vel, sample_actions_series)  
        iva_dataloader = DataLoader(dataset=iva_dataset, batch_size=config.plan_batch_size, shuffle=False, num_workers=2, drop_last=False)
        
        # print('start propagating action samples...')
        time.sleep(0.5)
        out_list = []
        
        self.model.eval()
        with torch.no_grad():
            # for batch_image_tensor, batch_vel, batch_action_series in tqdm(iva_dataloader):
            for batch_image_tensor, batch_vel, batch_action_series in iva_dataloader:
                batch_image_tensor = batch_image_tensor.to(self.device)
                batch_vel = batch_vel.to(self.device)
                batch_action_series = batch_action_series.to(self.device)

                output = self.model(batch_image_tensor, batch_vel, batch_action_series)
                out_list.append(output)
        

        new_actions = cal_optim_action_series(torch.cat(out_list,dim=0), torch.FloatTensor(pos).to(self.device), goal, sample_actions_series.to(self.device))
        # print('done.')
        
        return new_actions.cpu().numpy()

        
        
        
        


