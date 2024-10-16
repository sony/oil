import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from online.envs.bidding_env import BiddingEnv
from definitions import ROOT_DIR
import json

df = pd.read_parquet("/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/output/testing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023/dataset.parquet")

import pandas as pd
import numpy as np

def transform_dataframe(df):
    new_data = {
        'norm_obs': [],
        'pvalue': [],
        'pvalue_sigma': [],
        'oracle_action': [],
        'episode': [],
        'step': []
    }

    for _, row in df.iterrows():
        # Extract row-specific data
        norm_obs = row['norm_obs']
        pvalues = row['pvalues']
        pvalues_sigma = row['pvalues_sigma']
        oracle_action = row['oracle_action']
        episode = row['episode']
        step = row['step']

        # Iterate through each pvalue
        for i in range(len(pvalues)):
            # Append norm_obs, pvalue, pvalue_sigma and oracle_action
            new_data['norm_obs'].append(norm_obs)
            new_data['pvalue'].append(pvalues[i])
            new_data['pvalue_sigma'].append(pvalues_sigma[i])
            new_data['oracle_action'].append(oracle_action[i])
            new_data['episode'].append(episode)
            new_data['step'].append(step)

    # Create new transformed DataFrame
    transformed_df = pd.DataFrame(new_data)
    return transformed_df

# Assuming your DataFrame is called `df`
transformed_df = transform_dataframe(df)

from sklearn.model_selection import train_test_split

# Split by episode
unique_episodes = transformed_df['episode'].unique()

# Split into 80% train and 20% test
train_episodes, test_episodes = train_test_split(unique_episodes, test_size=0.2, random_state=42)

# Create train and test sets based on the episode split
train_df = transformed_df[transformed_df['episode'].isin(train_episodes)]
test_df = transformed_df[transformed_df['episode'].isin(test_episodes)]

import torch
from torch.utils.data import Dataset

class BiddingDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the row
        row = self.data.iloc[idx]

        # Prepare the input (norm_obs + pvalue + pvalue_sigma)
        norm_obs = np.array(row['norm_obs'], dtype=np.float32)
        pvalue = np.array(row['pvalue'], dtype=np.float32)
        pvalue_sigma = np.array(row['pvalue_sigma'], dtype=np.float32)

        # Concatenate inputs
        input_features = np.concatenate([norm_obs, [pvalue], [pvalue_sigma]])

        # Oracle action as the target
        target = np.array(row['oracle_action'], dtype=np.float32)

        # Convert to PyTorch tensors
        return torch.tensor(input_features), torch.tensor(target)

# Create PyTorch datasets
train_dataset = BiddingDataset(train_df)
test_dataset = BiddingDataset(test_df)

from torch.utils.data import DataLoader

# Set up DataLoader for batching
batch_size = 128  # Example batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class BidRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(BidRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # Output is a single value (the oracle action)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the model, optimizer, and loss function
input_size = 60 + 2  # norm_obs length + pvalue + pvalue_sigma
model = BidRegressionModel(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # You could also experiment with HuberLoss

# Training loop
def train_model(model, train_loader, test_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print training loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                test_loss += loss.item()

        print(f'Test Loss: {test_loss/len(test_loader)}')

# Train the model
train_model(model, train_loader, test_loader)
