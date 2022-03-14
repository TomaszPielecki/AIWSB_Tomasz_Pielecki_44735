import pandas as pd
from sklearn.model_selection import  train_test_split

df = pd.read_csv("D:\AI\iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=51182)

def classify_iris(s1, sw, pl, pw):
    if s1<=5.8 and sw>=2.3 and pl<=2 and pw<=1:
        return ("Setosa")
    elif 5<=s1<=7 and sw<=3.4 and 5.1>=pl>=3 and pw >=1 and pw<=1.8:
        return ("Versicolor")
    else:
        return ("Virginica")

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[i, 0],test_set[i, 1], test_set[i, 2], test_set[i,3]) == test_set[i, 4]:
        good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions/len*100,"%")