import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re

#Step 1 - General Model Training

class FakeNewsModel(nn.Module):
    """Defining the model architecture class
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(FakeNewsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class UrduDataset(Dataset):
    """Defining the class to represent the dataset
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = self.load_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        #load & preprocess the text data
        text_data = self.load_text_data(sample_path)
        return text_data, label

    def load_dataset(self):
        """domains, represented by text docs' by name
        e.g. technology = tc, sports = sp, showbiz = sbz, hlth = health, bs = business
        """
        samples = []
        domains = ["tc", "sp", "sbz", "hlth", "bs"]
        for domain in domains:
            domain_dir = os.path.join(self.root_dir, domain)
            for filename in os.listdir(domain_dir):
                if filename.endswith(".txt"):
                    #using REs to extract the domain from the filename
                    match = re.match(rf"{domain}(.+)\.txt", filename)
                    if match:
                        domain_extension = match.group(1)
                        filepath = os.path.join(domain_dir, filename)
                        label = 1 #fake label
                        samples.append((filepath, label, domain_extension))
        return samples

    def load_text_data(self, filepath):
        """Reading the text data from the file
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            text_data = file.read()
        return text_data

def sample_tasks(urdu_dataset, num_tasks=5):
    """Sample the tasks
    """
    tasks = []
    for _ in range(num_tasks):
        task_samples = urdu_dataset.samples[:10]  # Sample 10 samples for each task
        tasks.append(task_samples)
    return tasks

def split_task(task):
    """Split the tasks - Dsd (support set), Dqd (query set)
    """
    split_idx = len(task) // 2
    Dsd = task[:split_idx]
    Dqd = task[split_idx:]
    return Dsd, Dqd

#Initialize base model
input_size = 438  #input size
hidden_size = 70
output_size = 2  #2 classification outputs (fake or real)
base_model = FakeNewsModel(input_size, hidden_size, output_size)

#provide directory for dataset
urdu_dataset = UrduDataset(root_dir="C:\Documents\Training Dataset@FIRE2021\Train\Fake")

learning_rate_beta = 0.001
#optimizer for meta-learning
meta_optimizer = optim.SGD(base_model.parameters(), lr=learning_rate_beta)

num_meta_iterations = 1000 #iterations for meta-learning

for meta_iteration in range(num_meta_iterations):
    #sample a batch of training tasks
    tasks = sample_tasks(urdu_dataset)

    #now, update model parameters for each task
    for task in tasks:
        Dsd, Dqd = split_task(task)

        #convert data to tensors using pytorch function
        Dsd_data, Dsd_labels = zip(*Dsd)
        Dsd_data = torch.tensor(Dsd_data)
        Dsd_labels = torch.tensor(Dsd_labels)

        Dqd_data, Dqd_labels = zip(*Dqd)
        Dqd_data = torch.tensor(Dqd_data)
        Dqd_labels = torch.tensor(Dqd_labels)

        #perform trraining on support set (Dsd)
        optimizer = optim.SGD(base_model.parameters(), lr=learning_rate_alpha)
        for data, labels in zip(Dsd_data, Dsd_labels):
            optimizer.zero_grad()
            outputs = base_model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        #evaluation on query set (Dqd)
        for data, labels in zip(Dqd_data, Dqd_labels):
            outputs = base_model(data)
            loss = F.cross_entropy(outputs, labels)

        #update model parameters
        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()

#base model parameters, θ, updated after the iterations
