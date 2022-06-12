import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

df = pd.read_csv('../input/fer2013/fer2013/fer2013.csv')
df.head()

x_train = []
y_train = []
x_test = []
y_test = []


def preprocess():
    for index, row in df.iterrows():
        k = row['pixels'].split(" ")
        if row['Usage'] == 'Training':
            x_train.append(np.array(k))
            y_train.append(row['emotion'])
        elif row['Usage'] == 'PublicTest':
            x_test.append(np.array(k))
            y_train.append(row['emotion'])


preprocess()
print(x_train[0])
