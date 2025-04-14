import json
import pickle
import numpy as np
import pandas as pd
from pickle import dump
from os import listdir
from os.path import isfile, join
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

wandb.login()

class CustomTrafficDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FeedforwardNNModel(nn.Module):
    def __init__(self, idim, hdim, odim):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(idim, hdim, dtype=torch.float32), 
            nn.ReLU(),
            nn.Linear(hdim, hdim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hdim, odim, dtype=torch.float32),
            nn.Sigmoid()
        )  

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        pred = self.forward(x)
        return pred

class TrafficDataPoint:
    def __init__(self, ts, ids, added_car):
        self.ts = pd.to_datetime(ts, unit="s")
        self.ids = [car_id for car_id in ids]
        self.added = added_car

def fit(train_loader, model, opt, loss_fn, epochs = 1000, wandb = None):
  
  for _ in range(epochs):
    correct = 0
    total_entries = 0
    total_steps = 0
    mean_loss = 0

    for i, (features, labels) in enumerate(train_loader):
        opt.zero_grad()
        outputs = model(features)
        outputs = torch.reshape(outputs, (-1,))
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        opt.step()  

        predicted_labels = outputs.cpu().detach().numpy()
        predicted_labels = [1 if p > 0.5 else 0 for p in predicted_labels]
        expected_labels = labels.cpu().detach().numpy()
        correct += (predicted_labels == expected_labels).sum()
        total_entries += len(expected_labels)
        total_steps += 1
        mean_loss += loss.item()

    accuracy = 100 * correct / total_entries
    mean_loss /= total_steps
    if wandb is not None:
        wandb.log({"accuracy": accuracy, "loss": mean_loss})
    
  return loss.item()

def train_nn_model(features, labels):
    lr = 0.001
    epochs = 10


    wandb.init(
        project="TrafficStats",
        name="FFNN Traffic Generation 10April Sixth",
        config={                      
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    X_train, X_test, y_train, y_test = map(torch.tensor, train_test_split(features, labels, test_size=0.2))
    device = torch.device("cuda")
    X_train = X_train.float()
    X_train=X_train.to(device)
    y_train = y_train.long()
    y_train=y_train.to(device)

    train_ds = CustomTrafficDataset(X_train, y_train)
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)

    fn = FeedforwardNNModel(6, 12, 1)
    fn.to(device)

    loss_fn = nn.BCELoss()
    opt = optim.SGD(fn.parameters(), lr=lr)
    
    print(y_test)
    last_loss = fit(train_dataloader, fn, opt, loss_fn, epochs=epochs, wandb=wandb)
    print("Last loss: ", last_loss)

    torch.save(fn.state_dict(), "./models/NN/model_C5_10_aprilie_6")
    
    X_test = X_test.float()
    X_test = X_test.to(device)
    y_test = y_test.long()
    y_test = y_test.to(device)

    outputs = fn(X_test)
    outputs = torch.reshape(outputs, (-1,))
    loss_test = loss_fn(outputs, y_test.float())
    print("Test loss: ", loss_test)

    wandb.finish()

# Step 4: Main execution
if __name__ == "__main__":
    # one_week_data, intervals = read_data_from_files()
    # features, labels = prepare_data(one_week_data, intervals)
    with open('C5_final_traffic_dataset.pkl', 'rb') as f:
        feat_labels = pickle.load(f)

    features = [f[:(len(f) - 1)] for f in feat_labels]
    labels = [f[-1] for f in feat_labels]
    train_nn_model(features, labels)