import pandas as pd
from pickle import load
from train_nn import FeedforwardNNModel
import torch
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn

# if __name__ == "__main__":
#     with open("./models/C1/c1dt.pkl", "rb") as f:
#         clf = load(f)


#     cars = []
#     for t in pd.date_range(pd.Timestamp('2020-11-15 22:06:16'), pd.Timestamp('2020-11-22 22:07:16'), freq="3s"):
#         d = t.day_of_week
#         h = t.hour
#         is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
#         is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
#         is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
#         is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21
#         num_cars = len(cars)

#         to_add = clf.predict_proba([[d, is_night, is_morning, is_day, is_evening, num_cars]])
#         print(to_add)
#         if to_add[0][1] == 1:
#             print("Added")
#             cars.append(max(cars) + 1)

if __name__ == "__main__":
    with open("c5lr.pkl", "rb") as f:
        logistic_regression = load(f)
    
    # Decision Tree
    with open("c5dt.pkl", "rb") as f:
        decision_tree = load(f)
    # Random Forest Classifier
    with open("c5rf.pkl", "rb") as f:
        random_forest = load(f)

    traffic = [[] for _ in range(24)]
    week_traffic = [[[] for _ in range(24)] for _ in range(7)]

    prob_array = []
    max_num_cars = -1
    num_cars = 1
    cnt = 0
    for t in pd.date_range(pd.Timestamp('2025-03-23 00:00:00'), pd.Timestamp('2025-03-29 23:59:00'), freq="3s"):
        d = t.day_of_week
        h = t.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21

        lr_prediction = logistic_regression.predict([[d, is_night, is_morning, is_day, is_evening, num_cars]])

        if lr_prediction:
            num_cars += 1

        if num_cars > max_num_cars:
            max_num_cars = num_cars
        
        if cnt == 20:
            if h > 23 or h < 2:
                num_cars = 0
            else:
                num_cars = 1
            cnt = 0

        cnt += 1
        
        traffic[h].append(num_cars)
        week_traffic[d][h].append(num_cars)

    print(max_num_cars)
    for h in range(24):
        if len(traffic[h]) > 0:
            traffic[h] = np.median(traffic[h])
        else:
            traffic[h] = 0

        for d in range(7):
            if len(week_traffic[d][h]) > 0:
                week_traffic[d][h] = (np.median(week_traffic[d][h]), d >= 5)
            else:
                week_traffic[d][h] = (0, d >= 5)

    week_traffic_array = []
    for d in range(7):
        for h in range(24):
            week_traffic_array.append(week_traffic[d][h])

    df = pd.DataFrame.from_dict({"ts": [i for i in range(0, len(traffic))], "traffic": traffic})
    week_df = pd.DataFrame.from_dict({"ts": [i for i in range(len(week_traffic_array))], "traffic": [t[0] for t in week_traffic_array], "weekend": [t[1] for t in week_traffic_array]})

    sns.lineplot(data=week_df, x="ts", y="traffic")

    plt.show()


    traffic = [[] for _ in range(24)]
    week_traffic = [[[] for _ in range(24)] for _ in range(7)]

    prob_array = []
    max_num_cars = -1
    num_cars = 1
    cnt = 0
    for t in pd.date_range(pd.Timestamp('2025-03-23 00:00:00'), pd.Timestamp('2025-03-29 23:59:00'), freq="3s"):
        d = t.day_of_week
        h = t.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21

        dt_prediction = decision_tree.predict([[d, is_night, is_morning, is_day, is_evening, num_cars]])

        if dt_prediction:
            num_cars += 1

        if num_cars > max_num_cars:
            max_num_cars = num_cars
        
        if cnt == 20:
            if h > 23 or h < 2:
                num_cars = 0
            else:
                num_cars = 1
            cnt = 0

        cnt += 1
        
        traffic[h].append(num_cars)
        week_traffic[d][h].append(num_cars)

    print(max_num_cars)
    for h in range(24):
        if len(traffic[h]) > 0:
            traffic[h] = np.median(traffic[h])
        else:
            traffic[h] = 0

        for d in range(7):
            if len(week_traffic[d][h]) > 0:
                week_traffic[d][h] = (np.median(week_traffic[d][h]), d >= 5)
            else:
                week_traffic[d][h] = (0, d >= 5)

    week_traffic_array = []
    for d in range(7):
        for h in range(24):
            week_traffic_array.append(week_traffic[d][h])

    df = pd.DataFrame.from_dict({"ts": [i for i in range(0, len(traffic))], "traffic": traffic})
    week_df = pd.DataFrame.from_dict({"ts": [i for i in range(len(week_traffic_array))], "traffic": [t[0] for t in week_traffic_array], "weekend": [t[1] for t in week_traffic_array]})
    sns.lineplot(data=week_df, x="ts", y="traffic")

    plt.show()

    traffic = [[] for _ in range(24)]
    week_traffic = [[[] for _ in range(24)] for _ in range(7)]

    prob_array = []
    max_num_cars = -1
    num_cars = 1
    cnt = 0
    for t in pd.date_range(pd.Timestamp('2025-03-25 00:00:00'), pd.Timestamp('2025-03-25 23:59:00'), freq="3s"):
        d = t.day_of_week
        h = t.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21

        rf_prediction = random_forest.predict([[d, is_night, is_morning, is_day, is_evening, num_cars]])

        if rf_prediction:
            num_cars += 1

        if num_cars > max_num_cars:
            max_num_cars = num_cars
        
        if cnt == 20:
            if h > 23 or h < 2:
                num_cars = 0
            else:
                num_cars = 1
            cnt = 0

        cnt += 1
        
        traffic[h].append(num_cars)
        week_traffic[d][h].append(num_cars)

    print(max_num_cars)
    for h in range(24):
        if len(traffic[h]) > 0:
            traffic[h] = np.median(traffic[h])
        else:
            traffic[h] = 0

        for d in range(7):
            if len(week_traffic[d][h]) > 0:
                week_traffic[d][h] = (np.median(week_traffic[d][h]), d >= 5)
            else:
                week_traffic[d][h] = (0, d >= 5)

    week_traffic_array = []
    for d in range(7):
        for h in range(24):
            week_traffic_array.append(week_traffic[d][h])

    df = pd.DataFrame.from_dict({"ts": [i for i in range(0, len(traffic))], "traffic": traffic})
    week_df = pd.DataFrame.from_dict({"ts": [i for i in range(len(week_traffic_array))], "traffic": [t[0] for t in week_traffic_array], "weekend": [t[1] for t in week_traffic_array]})

    sns.lineplot(data=week_df, x="ts", y="traffic")
    plt.show()
