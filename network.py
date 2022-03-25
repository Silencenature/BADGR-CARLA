import torch
from torch import nn

class PredictNetwork(nn.Module):
    def __init__(self):
        super(PredictNetwork, self).__init__()

        self._obs_im_model = nn.Sequential(
            *[
                nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2), bias=False),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), bias=False),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(289536, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            ]
        )

        self._obs_vec_model = nn.Sequential(
            *[
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            ]
        )

        self._obs_lowd_model = nn.Sequential(
            *[
                nn.Linear(160, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            ]
        )

        self._action_model = nn.Sequential(
            *[
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
            ]
        )
        
        self.lstm = nn.LSTM(16,64,batch_first = True)

        self._output_model = nn.Sequential(
            *[
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
            ]
        )
        
        self._sigmoid = nn.Sigmoid()

    def cal_lstm(self, initial_state, action_series):
        initial_state = initial_state.unsqueeze(0)
        initial_state_c, initial_state_h = initial_state.chunk(2, dim=2)
        hidden = (initial_state_h.contiguous(), initial_state_c.contiguous())
        
        lstm_out, _ = self.lstm(action_series,hidden)
        # return lstm_out[:,-1,:]
        return lstm_out

    def forward(self, image, vel, action_series):
        image_lowd = self._obs_im_model(image)
        vel_lowd = self._obs_vec_model(vel)
        obs_lowd = self._obs_lowd_model(torch.cat((image_lowd,vel_lowd),1))

        actions_lowd = []
        for i in range(10):
            actions_lowd.append(self._action_model(action_series[:,i,:].squeeze(1)))
         
        actions_lowd = torch.stack(actions_lowd,dim=1)

        rnn_out = self.cal_lstm(obs_lowd, actions_lowd)

        output = []
        for i in range(10):
            temp = self._output_model(rnn_out[:,i,:])
            temp[:2] = self._sigmoid(temp[:2])
            output.append(temp)

        return torch.stack(output,dim=1)
    
