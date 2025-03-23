import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    accuracies = [0.9999504607153473, 0.9994324631101021, 0.9994324631101021, 0.9994324631101021, 0.9994324631101021]
    precisions = [0.9991539763113367, 0.9994324631101021, 0.9994324631101021, 0.9994324631101021, 0.9994324631101021]
    cases = [f"C{c}" for c in range(1, 6)]
    
    df_map = {"cases": cases, "acc": accuracies, "prec": precisions}
    df = pd.DataFrame.from_dict(df_map)
    df = df.melt("cases", var_name="measures", value_name="values")
    print(df)
    sns.catplot(data=df, x="cases", y="values", hue="measures", kind="point")
    plt.show()

    df_map = {"cases": cases, "real_neg_points": [570338, 448772, 29918, 22436, 14959], "real_poz_points": [34462, 34462, 34462, 34462, 34462], "covered_poz_points": [0, 156028, 156028, 156028, 156028]}
    df = pd.DataFrame.from_dict(df_map)
    df = df.melt("cases", var_name="point_type", value_name="point_count")

    sns.catplot(data=df, x="cases", y="point_count", hue="point_type", kind="point")
    plt.show()