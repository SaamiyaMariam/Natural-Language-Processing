import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, decoders
import matplotlib.pyplot as plt
from transferability import *

#Step 3 - Target Domain Adaptation

class GeneralFakeNewsModel(nn.Module):
    """Defining the fake news detection model
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(GeneralFakeNewsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_source_instance_weights(urdu_source_dataset):
    """obtaining source instance weights
    using transferability indicators in Step 2
    """
    return [1.0] * len(urdu_source_dataset)

def get_target_instance_weights(inputs):
    """obtaining target instance weights
    """
    return torch.rand(inputs.size(0))

def reweigh_source_samples(urdu_source_dataset, source_instance_weights):
    """Re-weighing source samples
    """
    reweighed_samples = []
    for (text_data, label), weight in zip(urdu_source_dataset.samples, source_instance_weights):
        reweighed_samples.extend([(text_data, label)] * int(weight * 100))  #convert to integers
    return reweighed_samples

def combine_samples(reweighed_source_samples, inputs):
    """concatenating samples
    """
    combined_samples = reweighed_source_samples + [(text_data, label) for text_data, label in inputs]
    return combined_samples

#Initialize the general model
input_size = 438
hidden_size = 70
output_size = 2  #2 classification outputs (fake or real)
general_model = GeneralFakeNewsModel(input_size, hidden_size, output_size)

#optimizer for model adaptation
optimizer = torch.optim.Adam(general_model.parameters(), lr=0.001)

num_iterations = 1000 #iterations for model adaptation

source_instance_weights = get_source_instance_weights(UrduDataset())

for iteration in tqdm(range(num_iterations), desc="Target Domain Adaptation"):
    for batch in DataLoader(UrduDataset(), batch_size=32, shuffle=True):
        inputs, targets = preprocess_batch(batch, tokenizer)

        #transferability weights for target domain samples
        target_instance_weights = get_target_instance_weights(inputs)

        #Re-weigh source domain samples based on transferability weights
        reweighed_source_samples = reweigh_source_samples(UrduDataset(), source_instance_weights)

        #combine re-weighed source samples with target domain samples
        combined_samples = combine_samples(reweighed_source_samples, inputs)

        optimizer.zero_grad()
        outputs = general_model(combined_samples)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
