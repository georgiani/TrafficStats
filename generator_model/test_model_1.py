import pandas as pd
from pickle import load
from train_nn import FeedforwardNNModel
import torch
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
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
    fn = FeedforwardNNModel(6, 12, 2)
    fn.load_state_dict(torch.load("./models/NN/model_C5_8_aprilie_1", weights_only=True))
    fn.eval()

    traffic = [[] for _ in range(24)]
    week_traffic = [[[] for _ in range(24)] for _ in range(7)]


    prob_array = []
    max_num_cars = -1
    num_cars = 1
    cnt = 0
    for t in pd.date_range(pd.Timestamp('2025-03-25 00:00:00'), pd.Timestamp('2025-03-30 23:59:00'), freq="3s"):
        d = t.day_of_week
        h = t.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21

        probs = fn.predict(torch.Tensor([d, is_night, is_morning, is_day, is_evening, num_cars]))

        probs = probs.detach().numpy()
        prob_value = np.argmax(probs)
        prob_array.append(prob_value) 
        if prob_value == 1:
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
                week_traffic[d][h] = np.median(week_traffic[d][h])
            else:
                week_traffic[d][h] = 0

    week_traffic_array = []
    for d in range(7):
        for h in range(24):
            week_traffic_array.append(week_traffic[d][h])

    print(len(traffic))
    print(len(week_traffic_array))

    df = pd.DataFrame.from_dict({"ts": [i for i in range(0, len(traffic))], "traffic": traffic})
    week_df = pd.DataFrame.from_dict({"ts": [i for i in range(len(week_traffic_array))], "traffic": week_traffic_array})
    # sns.lineplot(data=df, x="ts", y="traffic")
    sns.lineplot(data=week_df, x="ts", y="traffic")

    plt.show()
