import numpy as np
from envs.simulator import Agent
import pickle
import time
from collections import deque

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   


class Noise(object):
    def __init__(self, theta=150, mu_steer=-0.5, mu_throttle=0.2, mu_break=0.0, sigma=3, n_steps_annealing=100):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        
        self.x0_steer = 0
        self.x0_throttle = 0
        self.x0_break = 0
        
        self.mu_steer = mu_steer
        self.mu_throttle = mu_throttle
        self.mu_break = mu_break

    def get_action(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        
        self.x0_steer = self.x0_steer + self.theta * (self.mu_steer - self.x0_steer) * 0.01 + sigma * 0.1 * np.random.normal()
        self.x0_throttle = self.x0_throttle + self.theta * (self.mu_throttle - self.x0_throttle) * 0.01 + sigma * 0.1 * np.random.normal()
        self.x0_break = self.x0_break + self.theta * (self.mu_break - self.x0_break) * 0.01 + sigma * 0.1 * np.random.normal()
        
        return [np.clip(self.x0_steer,-1,1), np.clip(self.x0_throttle,0.0,1.0), np.clip(self.x0_break,0.0,1.0)]
        # return [np.clip(self.x0_steer,-1,1), np.clip(self.x0_throttle,0.0,1.0), 0]               


# def process_label(lane, coll, pos):
#     if lane:
#         lane = np.array([1])
#     else:
#         lane = np.array([0])
    
#     if coll:
#         coll = np.array([1])
#     else:
#         coll = np.array([0])

#     return np.concatenate((lane,coll,pos))

def process_label(lane, coll, pos):
    if lane:
        lane = [1]
    else:
        lane = [0]
    
    if coll:
        coll = [1]
    else:
        coll = [0]

    return lane+coll+pos


def save_data(image, vel, action_series, label_series, frame_idx):
    pack = {'image':image, 'vel':vel, 'action_series':np.array(action_series), 'label':np.array(label_series)}
    file_name = '.\\dataset\\%s_%05d.pkl' % (time.strftime("%m-%d-%H-%M-%S", time.localtime()), (frame_idx+50180))
    f_save = open(file_name, 'wb')
    pickle.dump(pack, f_save)
    f_save.close()


def main():
    agent = Agent()
    add_noise = Noise()

    # Start collecting data
    frames = 10000
    frame_idx = 1
    max_steps = 100
    episode = 0
    H = 10

    while frame_idx < frames:
        episode += 1
        agent.reset()
        action = [0.0, 0.0, 0.0]
        while True:
            image, vel, lane, coll, pos, done = agent.step(action)
            if vel[2] == 0.0:
                break
        
        image_deque = deque()
        vel_deque = deque()
        action_list = []
        label_list = []
        step = 0
        while not done and step < max_steps:
            while step < H:
                image_deque.append(image)
                vel_deque.append(vel)
                action_list.append(action)
                label_list.append(process_label(lane, coll, pos))
                step += 1
                action = add_noise.get_action(step)
                for i in range(2):
                    image, vel, lane, coll, pos, done = agent.step(action)
                    if done:
                        break
            
            if done:
                episode -= 1
                print('fail in %d steps' % H)
                break

            image_save = image_deque.popleft()
            vel_save = vel_deque.popleft()
            action_series_save = action_list[:H]
            action_list.pop(0)
            label_series_save = label_list[:H]
            label_list.pop(0)
            save_data(image_save, vel_save, action_series_save, label_series_save, frame_idx)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print('%d%% [%s / %s]' % (frame_idx/frames*100, frame_idx, frames))

            image_deque.append(image)
            vel_deque.append(vel)
            action_list.append(action)
            label_list.append(process_label(lane, coll, pos))
            step += 1
            action = add_noise.get_action(step)
            for i in range(2):
                image, vel, lane, coll, pos, done = agent.step(action)
        
        if step >= H:
            image_save = image_deque.popleft()
            vel_save = vel_deque.popleft()
            action_series_save = action_list[:H]
            # action_list.pop(0)
            label_series_save = label_list[:H]
            # label_list.pop(0)
            save_data(image_save, vel_save, action_series_save, label_series_save, frame_idx)
            frame_idx += 1
            print('episode %d kept %.1fs' % (episode, 0.2*step))
    
    print('finish collecting data')
    agent.close()
    return


if __name__ == '__main__':
    main() 