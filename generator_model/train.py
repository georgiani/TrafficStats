import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import json

class TrafficDataPoint:
    def __init__(self, ts, ids, added_car):
        self.ts = pd.to_datetime(ts, unit="s")
        self.ids = [id for id in ids]
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

    for f in files:

        prev = None
        with open(f"date/{f}_fixed.json", "r") as res:
            res = json.load(res)
            df = pd.DataFrame(res)


            for c in df.columns:
                
                # If it's the first, then no added
                if prev is None:
                    added = False
                else:
                    # If it's not the first, the check if there is any new
                    #   ids in the current timestamp in comparison with the previous one
                    previous_ids = df[prev]["ids"]
                    current_ids = df[c]["ids"]

                    for current_id in current_ids:
                        if current_id not in previous_ids:
                            added = True
                            break
                
                data_point = TrafficDataPoint(df[c]["ts"], df[c]["ids"], added)
                full_data.append(data_point)

                prev = c

    return full_data

def data_point_not_in_data(data, d, h, m, s):
    for i in data:
        if i.ts.day_of_week == d and i.ts.hour == h and i.ts.minute == m and i.ts.second == s:
            return False

    return True 

def prepare_data(data):
    features = []
    labels = []

    # Data we have
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
        num_cars = len(i.ids)

        features.append([day, is_night, is_morning, is_day, is_evening, num_cars])
        labels.append(i.added)

        covered_time_points[i.ts.day_of_week][i.ts.hour][i.ts.minute][i.ts.second] = True

    # Data for the missing parts
    for d in range(0, 7):
        for h in range(0, 24):
            for m in range(0, 60):
                for s in range(0, 60):
                    if covered_time_points[d][h][m][s] == False:
                        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
                        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
                        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
                        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21
                        num_cars = 0

                        features.append([day, is_night, is_morning, is_day, is_evening, num_cars])
                        labels.append(False)
                        covered_time_points[d][h][m][s] = True

    return np.array(features), np.array(labels)

# Step 3: Train the model
def train_model(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Logistic Regression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred_lr = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("Logistic Regression Precision:", precision_score(y_test, y_pred_lr))

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    y_pred_dt = decision_tree.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
    print("Decision Tree Precision:", precision_score(y_test, y_pred_dt))

    # Random Forest Classifier
    random_forest = RandomForestClassifier(max_depth=2, random_state=0)
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Decision Tree Precision:", precision_score(y_test, y_pred_rf))

# Step 4: Main execution
if __name__ == "__main__":
    one_week_data = read_data_from_files()
    features, labels = prepare_data(one_week_data)
    print(features)
    train_model(features, labels)