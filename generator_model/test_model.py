import pandas as pd
from pickle import load
from train_nn import FeedforwardNNModel
import torch
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

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
    fn.load_state_dict(torch.load("./models/NN/model", weights_only=True))
    fn.eval()

    ts = [0]
    traffic = [[[] for _ in range(24)] for _ in range(7)]

    cars = []
    cnt = 0
    for t in pd.date_range(pd.Timestamp('2025-03-24 08:00:00'), pd.Timestamp('2025-03-30 23:59:00'), freq="3s"):
        d = t.day_of_week
        h = t.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21
        num_cars = len(cars)

        probs = fn.predict(torch.Tensor([d, is_night, is_morning, is_day, is_evening, num_cars]))
        to_add = np.argmax(probs.detach().numpy()) == 0
        if to_add:
            if len(cars) == 0:
                cars.append(1)
            else:
                cars.append(max(cars) + 1)

        cnt += 1

        if cnt == 20:
            cars.clear()
            cnt = 0
        
        traffic[d][h].append(len(cars))


    mean_traffic = []
    for d in range(0, 7):
        for h in range(0, 24):
            mean_traffic.append(np.mean(traffic[d][h]))

    df = pd.DataFrame.from_dict({"ts": [i for i in range(0, len(mean_traffic))], "traffic": mean_traffic})
    sns.lineplot(data=df, x="ts", y="traffic")

    plt.show()
