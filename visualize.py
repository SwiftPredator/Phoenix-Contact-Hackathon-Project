import matplotlib.pyplot as plt
import pandas as pd


def visualize_rooms(df: pd.DataFrame):
    dfg = df.groupby("Room")
    dfg.plot(kind="line", x="Timestamp", y="RoomTemperature")

    plt.show()
