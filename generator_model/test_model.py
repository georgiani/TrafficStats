import pandas as pd
from pickle import load

if __name__ == "__main__":
    with open("./models/C1/c1dt.pkl", "rb") as f:
        clf = load(f)


    cars = []
    for t in pd.date_range(pd.Timestamp('2020-11-15 22:06:16'), pd.Timestamp('2020-11-22 22:07:16'), freq="3s"):
        d = t.day_of_week
        h = t.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21
        num_cars = len(cars)

        to_add = clf.predict_proba([[d, is_night, is_morning, is_day, is_evening, num_cars]])
        print(to_add)
        if to_add[0][1] == 1:
            print("Added")
            cars.append(max(cars) + 1)