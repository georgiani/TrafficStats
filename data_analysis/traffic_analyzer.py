import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(14, 6), dpi=100)

files = [
    "res_tuesday_1", 
    "res_tuesday_2", 
    "res_wed_0", 
    "res_wed_1", 
    "res_wed_2", 
    "res_wed_3", 
    "res_wed_4", 
    "res_thu_0", 
    "res_thu_1", 
    "res_thu_2", 
    "res_thu_3_night",
    "res_friday_0",
    "res_friday_1",
    "res_saturday_0",
    "res_sunday_1", 
    "res_sunday_2"
]


# [[[[0 ..,46, 47 ..],46, 47 ..],22, 23 ..], tuesday, wednesday, ...]

week_data = []
for _ in range(0, 7):
    day_data = []
    for _ in range(0, 24):
        hour_data = []

        for _ in range(0, 60):
            minute_data = [[0 for _ in range(2)] for _ in range(0, 60)]
            hour_data.append(minute_data)

        day_data.append(hour_data)

    week_data.append(day_data)

if __name__ == "__main__":
    for f in files:
        with open(f"date/{f}_fixed.json", "r") as res:
            res = json.load(res)
            df = pd.DataFrame(res)

            for c in df.columns:
                t = pd.to_datetime(df[c]['ts'], unit='s')
                cars = df[c]["ids"]
                number_of_cars = len(cars)
                week_data[t.day_of_week][t.hour][t.minute][t.second][0] += number_of_cars
                week_data[t.day_of_week][t.hour][t.minute][t.second][1] += 1

    day_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    color = ["red", "green", "blue", "yellow", "black", "red", "green"]

    SETTING = "Hour"

    i = 0
    x = []
    y = []
    ticks = []
    labels = []
    colors = []

    # By Minute
    if SETTING == "Second":
        for w in range(0, 7):
            ticks.append(i)
            labels.append(day_of_the_week[w])
            for h in range(0, 24):
                for m in range(0, 60):
                    for s in range(0, 60):
                        current_data = week_data[w][h][m][s]
                        number_of_cars = float(current_data[0])
                        number_of_data_points = current_data[1]

                        if number_of_cars != 0 and number_of_data_points != 0: 
                            x.append(i)
                            y.append(number_of_cars / number_of_data_points)

                            i += 1

        plt.plot(x, y)
        plt.xticks(ticks, labels)

        y_min = min(y)
        for i, t in enumerate(ticks):
            if i + 1 != len(ticks):
                plt.fill_between([ticks[i], ticks[i + 1]], y_min, 50, alpha=0.2)
            else:
                plt.fill_between([ticks[i], x[-1]], y_min, 50, alpha=0.2)


        plt.show()
    elif SETTING == "Minute":
        for w in range(0, 7):
            ticks.append(i)
            labels.append(day_of_the_week[w])
            for h in range(0, 24):
                for m in range(0, 60):
                    current_data = week_data[w][h][m]
                    number_of_cars = float(sum([current_data[m][0] for m in range(0, 60)]))
                    number_of_data_points = sum([current_data[m][1] for m in range(0, 60)])

                    if number_of_cars != 0 and number_of_data_points != 0: 
                        x.append(i)
                        y.append(number_of_cars / number_of_data_points)

                        i += 1
            
        # Normal plot
        # plt.plot(x, y)

        # Bar plot
        plt.bar(x, y)
        plt.xticks(ticks, labels)

        y_min = min(y)
        for i, t in enumerate(ticks):
            if i + 1 != len(ticks):
                plt.fill_between([ticks[i], ticks[i + 1]], y_min, 50, alpha=0.2)
            else:
                plt.fill_between([ticks[i], x[-1]], y_min, 50, alpha=0.2)


        plt.show()
    elif SETTING == "Hour":
        for w in range(0, 7):
            ticks.append(i)
            labels.append(day_of_the_week[w])
            for h in range(0, 24):
                number_of_cars = 0
                number_of_data_points = 0
                for m in range(0, 60):
                    current_data_minute = week_data[w][h][m]
                    number_of_cars_minute = float(sum([current_data_minute[m][0] for m in range(0, 60)]))
                    number_of_data_points_minute = sum([current_data_minute[m][1] for m in range(0, 60)])

                    number_of_cars += number_of_cars_minute
                    number_of_data_points += number_of_data_points_minute

                if number_of_cars != 0 and number_of_data_points != 0: 
                    x.append(i)
                    y.append(number_of_cars / number_of_data_points)

                    i += 1
                
            
        # Normal plot
        # plt.plot(x, y)

        # Bar plot
        plt.bar(x, y)
        plt.xticks(ticks, labels)

        for i, t in enumerate(ticks):
            if i + 1 != len(ticks):
                plt.fill_between([ticks[i], ticks[i + 1]], 0, 50, alpha=0.2)
            else:
                plt.fill_between([ticks[i], x[-1]], 0, 50, alpha=0.2)


        plt.show()