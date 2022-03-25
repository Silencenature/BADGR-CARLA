from torch.utils.data import Dataset

class IVA_Dataset(Dataset):
    def __init__(self, image, vel, action_series):
        import torchvision.transforms as transforms
        import torch

        self.transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                                            ])
        
        self.image = self.transforms(image)
        self.vel = torch.FloatTensor(vel)
        self.action_series = action_series

    def __getitem__(self, index): 
        return self.image, self.vel, self.action_series[index]

    def __len__(self):
        return len(self.action_series)