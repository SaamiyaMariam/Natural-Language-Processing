import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, decoders
import matplotlib.pyplot as plt
from domain_adaptation import *

#Step 4 - Evaluation

def train_ditfend_model(base_model_name=None, transfer_type=None, urdu_dataset=None):
    """DITFEND model
    """
    ditfend_model = GeneralFakeNewsModel(input_size, hidden_size, output_size)

    #optimizer for DITFEND model  using pytorch
    ditfend_optimizer = torch.optim.Adam(ditfend_model.parameters(), lr=0.001)

    num_iterations = 1000 #iterations for DITFEND model training

    for iteration in tqdm(range(num_iterations), desc="DITFEND Training"):
        for batch in DataLoader(urdu_dataset, batch_size=32, shuffle=True):
            inputs, targets = preprocess_batch(batch, tokenizer)

            target_instance_weights = get_target_instance_weights(inputs)

            reweighed_source_samples = reweigh_source_samples(UrduDataset(), source_instance_weights)

            combined_samples = combine_samples(reweighed_source_samples, inputs)

            ditfend_optimizer.zero_grad()
            outputs = ditfend_model(combined_samples)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            ditfend_optimizer.step()

    return ditfend_model

def evaluate_ditfend_model(ditfend_model, target_domain, urdu_dataset):
    """evaluating DITFEND model using dataset
    """
    ditfend_model.eval()

    #for evaluation metrics (e.g. accuracy)
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in DataLoader(urdu_dataset, batch_size=32, shuffle=False):
            inputs, targets = preprocess_batch(batch, tokenizer)

            #a forward pass
            outputs = ditfend_model(inputs)
            _, predicted_labels = torch.max(outputs, 1)

            #Update evaluation metrics
            correct_predictions += (predicted_labels == targets).sum().item()
            total_predictions += targets.size(0)

    accuracy = correct_predictions / total_predictions
    return {"accuracy": accuracy}

def create_mixed_dataset(urdu_dataset):
    """Creating a dataset from multiple domains to create a mixed dataset
    """
    mixed_samples = []
    for domain in urdu_dataset.domains:
        domain_samples = [sample for sample in urdu_dataset.samples if domain in sample[0]]
        mixed_samples.extend(domain_samples)
    return mixed_samples

#measuring effectiveness
transfer_types = ["domain", "instance", "both"]

for transfer_type in transfer_types:
    #train and evaluate DITFEND for each transfer type
    ditfend_model = train_ditfend_model(transfer_type=transfer_type, urdu_dataset=UrduDataset()) #iterate over each transfer type
    performance_metrics = evaluate_ditfend_model(ditfend_model, target_domain="health", urdu_dataset=UrduDataset())
    print(f"Performance of DITFEND with {transfer_type}-level transfer: {performance_metrics}")

#performance on dataset
mixed_dataset = create_mixed_dataset(UrduDataset())
ditfend_model = train_ditfend_model(urdu_dataset=mixed_dataset)
performance_metrics = evaluate_ditfend_model(ditfend_model, target_domain="mixed", urdu_dataset=mixed_dataset)
print(f"Performance of DITFEND on the mixed dataset: {performance_metrics}")