
# coding: utf-8

# In[127]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from skimage import io, transform
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# In[128]:


train_frame = pd.read_csv('hw7data/train.csv')

n = 0
img_id = train_frame.iloc[n, 1] + format('.jpg')
landmark_id = train_frame.iloc[n, -1]

print('Image name: {}'.format(img_id))
print('Landmark id: {}'.format(landmark_id))


# In[129]:


class LandmarksDataset(Dataset):
    """Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    #__len__ so that len(dataset) returns the size of the dataset.
    def __len__(self):
        return len(self.data)

    #__getitem__ to support the indexing such that dataset[i] can be used to get ith sample
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1] + '.jpg')
        image = io.imread(img_name)
        landmark_id = self.data.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)

        return image, landmark_id


# In[130]:


input_dataset = LandmarksDataset(csv_file='hw7data/train.csv',
                                    root_dir='hw7data/images', 
                                 transform=transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ]))


# In[131]:


dataloader = DataLoader(input_dataset, batch_size=50, shuffle=True, drop_last=True)
dataset_size = len(input_dataset)
class_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[137]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        scheduler.step()
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        batch_count = 0
        for inputs, labels in dataloader:
            batch_count += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                print('batch_count = {}, Loss = {}'.format(batch_count, loss))
                
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
                
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "E{}_weights.w".format(epoch))

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    #model.load_state_dict(torch.load('E1_weights.w'))
    return model


# In[138]:


model_ft = models.resnet18(pretrained=True)
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[139]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)


# In[141]:


class TestDataset(Dataset):
    """Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    #__len__ so that len(dataset) returns the size of the dataset.
    def __len__(self):
        return len(self.data)

    #__getitem__ to support the indexing such that dataset[i] can be used to get ith sample
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1] + '.jpg')
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image


# In[149]:


test_dataset = TestDataset(csv_file='hw7data/test.csv',
                                    root_dir='hw7data/images', 
                                 transform=transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ]))


# In[150]:


testdataloader = DataLoader(test_dataset, shuffle=False, drop_last=True)
test_dataset_size = len(test_dataset)
#class_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[163]:


def test(model, weight_file, test_dataloader):
    model.eval()
    model.load_state_dict(torch.load(weight_file))
    pred_labels = []
    for inputs in test_dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            pred_labels.append(preds.item())
    with open('submission.txt', 'w+') as f:
        f.write('landmark_id\n')
        for pred in pred_labels:
            f.write('{}\n'.format(pred))


# In[164]:


test(model_ft, 'E1_weights.w', testdataloader)

