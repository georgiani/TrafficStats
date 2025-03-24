import json
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
            nn.Softmax()
        )  

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        Y_pred = self.forward(x)
        return Y_pred

class TrafficDataPoint:
    def __init__(self, ts, ids, added_car):
        self.ts = pd.to_datetime(ts, unit="s")
        self.ids = [car_id for car_id in ids]
        self.added = added_car


covered_time_points = []
for _ in range(0, 7):
    day_data = []
    for _ in range(0, 24):
        hour_data = []

        for _ in range(0, 60):
            minute_data = [False for _ in range(0, 60)]
            hour_data.append(minute_data)

        day_data.append(hour_data)

    covered_time_points.append(day_data)

def read_data_from_files():
    full_data = []

    # files = [
    #     "res_monday_0",
    #     "res_monday_1",
    #     "res_tuesday_1", 
    #     "res_tuesday_2", 
    #     "res_wed_0", 
    #     "res_wed_1", 
    #     "res_wed_2", 
    #     "res_wed_3", 
    #     "res_wed_4", 
    #     "res_thu_0", 
    #     "res_thu_1", 
    #     "res_thu_2", 
    #     "res_thu_3_night",
    #     "res_friday_0",
    #     "res_friday_1",
    #     "res_saturday_0",
    #     "res_sunday_1", 
    #     "res_sunday_2"
    # ]
    files = [f"date/{f}" for f in listdir("./date") if isfile(join("./date", f))]

    intervals = []
    for f in files:
        prev_ids = None

        interval_min = 0
        interval_max = 0
        with open(f, "r") as res:
            res = json.load(res)

            res_keys = res.keys()
            df_preparation = {"idx": [k for k in res_keys], "ids": [res[k]["ids"] for k in res_keys], "ts" : [res[k]["ts"] for k in res_keys]}
            df = pd.DataFrame.from_dict(df_preparation)
            df = df.set_index("idx")

            interval_min, interval_max = df["ts"].iloc[0], df["ts"].iloc[-1]
            intervals.append((interval_min, interval_max))

            for _, row in df.iterrows():
                # If it's the first, then no added
                if prev_ids is None:
                    added = False
                else:
                    # If it's not the first, the check if there is any new
                    #   ids in the current timestamp in comparison with the previous one
                    previous_ids = prev_ids
                    current_ids = row["ids"]

                    for current_id in current_ids:
                        if current_id not in previous_ids:
                            added = True
                            break
                
                data_point = TrafficDataPoint(row["ts"], row["ids"], added)
                full_data.append(data_point)

                prev_ids = row["ids"]

    return full_data, intervals

def prepare_data(data, intervals):
    features = []
    labels = []

    # Data we have
    cars_added_count = 0
    added_count = 0
    for i in data:
        # Feature 1: Day
        day = i.ts.day_of_week

        # Feature 2: Part of day
        h = i.ts.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21

        # Feature 2: Number of cars in the previous state

        # Need to actually check if it's based on the number from previous state since
        #    this is the current number
        num_cars = len(i.ids)

        features.append([day, is_night, is_morning, is_day, is_evening, num_cars])
        labels.append(i.added)

        if i.added:
            cars_added_count += 1

        if covered_time_points[i.ts.day_of_week][i.ts.hour][i.ts.minute][i.ts.second] == False:
            added_count += 1 

        covered_time_points[i.ts.day_of_week][i.ts.hour][i.ts.minute][i.ts.second] = True
    
    print("Instances of added vs not ", cars_added_count, len(data) - cars_added_count)
    print("Really added time points ", added_count)

    # Cover everything related to intervals, in order to not add 0 at random in between correct data
    coverage_added = 0
    for i in intervals:
        min_range = pd.to_datetime(i[0], unit="s")
        max_range = pd.to_datetime(i[1], unit="s")

        for t in pd.date_range(min_range, max_range, freq="1s"):

            if covered_time_points[t.day_of_week][t.hour][t.minute][t.second] == False:
                coverage_added += 1
            covered_time_points[t.day_of_week][t.hour][t.minute][t.second] = True


    print("Coverage added ", coverage_added)
    # Data for the missing parts
    true_count = 0
    false_count = 0
    for d in range(0, 7):
        for h in range(0, 24):
            for m in range(0, 60):
                for s in range(0, 60):
                    if s % 15 == 0:
                        if covered_time_points[d][h][m][s] == False:
                            is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
                            is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
                            is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
                            is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21
                            num_cars = 0

                            features.append([d, is_night, is_morning, is_day, is_evening, num_cars])
                            labels.append(False)
                            covered_time_points[d][h][m][s] = True

                            false_count += 1

    print(false_count)
    print(true_count)
    print(len([a for a in labels if a == True]))
    print(len([a for a in labels if a == False]))
    return np.array(features), np.array(labels)


def fit(x, y, model, opt, loss_fn, epochs = 1000):
  
  for _ in range(epochs):
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    
  return loss.item()

def train_nn_model(features, labels):

    X_train, X_test, y_train, y_test = map(torch.tensor, train_test_split(features, labels, test_size=0.2))
    device = torch.device("cuda")

    print(X_train)

    X_train = X_train.float()
    y_train = y_train.long()

    print(y_train)

    X_train=X_train.to(device)
    y_train=y_train.to(device)
    fn = FeedforwardNNModel(6, 12, 2)
    fn.to(device)

    loss_fn = F.cross_entropy
    opt = optim.SGD(fn.parameters(), lr=0.5)

    print('Final loss', fit(X_train, y_train, fn, opt, loss_fn, epochs=10))

    X_test = X_test.float()
    X_test = X_test.to(device)
    y_test = y_test.long()
    y_test = y_test.to(device)
    print(loss_fn(fn(X_test), y_test).item())

    torch.save(fn.state_dict(), "./models/NN/model")

# Step 4: Main execution
if __name__ == "__main__":
    one_week_data, intervals = read_data_from_files()
    features, labels = prepare_data(one_week_data, intervals)
    train_nn_model(features, labels)