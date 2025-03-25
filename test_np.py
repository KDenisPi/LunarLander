from math import sqrt
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt


def distance(row):
    return sqrt(row[1]*row[1]+row[2]*row[2])

def test():

    df = ps.read_csv("data/10000_2.csv")
    x_data = df['Nm']

    np_arr = np.asarray(df)
    #print(np_arr)

    result = np.apply_along_axis(distance, 1, np_arr)
    #print(len(result))
    plt.plot(x_data, result)
    plt.grid(True)
    plt.savefig("distance.png")


if __name__ == '__main__':
    test()
