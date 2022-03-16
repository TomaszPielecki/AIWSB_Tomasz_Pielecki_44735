import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv("D:\AI\iris.csv")

print(df.dtypes)
print(df.describe())
df['petal.width'].plot.hist()
plt.show()
sns.pairplot(df, hue='variety')

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=51182)

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
wynik = dtc.score(test_inputs, test_classes)
print(wynik)