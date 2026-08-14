import csv

import matplotlib.pyplot as plt
import numpy
import pandas

csv_dir = "./training_loop/dqn/scalars"
paths = {"cav_path": f"{csv_dir}/chosen_action_value.csv",
         "rwd_path": f"{csv_dir}/reward.csv",
         "stp_reader": f"{csv_dir}/step_count.csv"}

data = []

for path in paths:
    with open(paths[path]) as csvfile:
        reader = pandas.read_csv(filepath_or_buffer=csvfile, usecols=[1,])
        data.append(numpy.array(reader).flatten())

plt.figure(0)
plt.plot(data[0])
plt.title("Average Chosen Action Value")

plt.figure(1)
plt.plot(data[1])
plt.title("Average Reward")

plt.figure(2)
plt.plot(data[2])
plt.title("Average Step Count")

plt.show()




