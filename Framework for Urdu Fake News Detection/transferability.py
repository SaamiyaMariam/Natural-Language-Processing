import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, processors, pre_tokenizers, decoders
import matplotlib.pyplot as plt
from model_training import *

#Step 2 - Transferability Quantifying

class LanguageModel(nn.Module):
    """Defining the language model architecture class
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.fc(output[:, -1, :])
        return output

def preprocess_batch(batch, tokenizer):
    """Preprocess a 'batch' (collection)
    """
    texts, targets = zip(*batch)

    #Tokenize the text
    encoded_batch = tokenizer.encode_batch(texts)
    inputs = torch.tensor([encoded.ids for encoded in encoded_batch])

    targets = torch.tensor(targets)

    return inputs, targets

def calculate_perplexity(model, inputs):
    """Calculating perplexity
    """
    outputs = model(inputs)
    probabilities = F.softmax(outputs, dim=1)
    perplexity = torch.exp(F.cross_entropy(outputs, targets))
    return perplexity.item()

#Initialize language model
vocab_size = 9480
embedding_dim = 50
hidden_dim = 100
output_dim = 2
target_model = LanguageModel(vocab_size, embedding_dim, hidden_dim, output_dim)

#optimizer for language model training
optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

num_iterations = 1000 #iterations for language model training

#using tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
tokenizer.train_from_iterator([text for text, _ in urdu_dataset.samples], trainer)

#Masked Language Modeling: training on the target domain
for iteration in tqdm(range(num_iterations), desc="Training Language Model"):
    for batch in DataLoader(urdu_dataset, batch_size=32, shuffle=True):
        inputs, targets = preprocess_batch(batch, tokenizer)
        optimizer.zero_grad()
        outputs = target_model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

#Calculating perplexity on target domain
val_dataset = urdu_dataset.samples[-int(len(urdu_dataset.samples) * 0.2):]  #20% as validation set
inputs, targets = preprocess_batch(val_dataset, tokenizer)
perplexity_target_domain = calculate_perplexity(target_model, inputs)

#transferability quantification for source instances
transferability_indicators = []

for source_article, label in tqdm(val_dataset, desc="Calculating Transferability"):
    masked_sentence, correct_word = generate_masked_sentence(source_article)
    encoded = tokenizer.encode(masked_sentence)
    inputs = torch.tensor(encoded.ids).unsqueeze(0)

    perplexity_source_instance = calculate_perplexity(target_model, inputs)
    transferability_indicator = 1 / perplexity_source_instance
    transferability_indicators.append(transferability_indicator)

#visualizing the distribution of transferability indicators
plt.hist(transferability_indicators, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title('Transferability Distribution')
plt.xlabel('Transferability Indicator')
plt.ylabel('Frequency')
plt.show()
