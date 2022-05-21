import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set()

df = pd.read_csv("D:\AI\iris.csv")

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=51182)

scaler = StandardScaler()
scaler.fit(train_inputs)

train_inputs = scaler.transform(train_inputs)
test_inputs = scaler.transform(test_inputs)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(train_inputs, train_classes)

y_pred = classifier.predict(test_inputs)

print(confusion_matrix(test_classes, y_pred))
print(classification_report(test_classes, y_pred))

# najlepszy wynik daje k = 3,5,11    k = 1 nie jest dokładne