from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()

datasets = train_test_split(iris.data, iris.target, test_size=0.3)

train_data, test_data, train_labels, test_labels = datasets

scaler = StandardScaler()

scaler.fit(train_data)


#1 ukryta warstwa i 2 neurony
mlp1 = MLPClassifier(hidden_layer_sizes=(2, ), max_iter=1000)

#1 ukryta warstwa i 3 neurony
mlp2 = MLPClassifier(hidden_layer_sizes=(3, ), max_iter=1000)

#2 ukryte warstwy, 3 neurony
mlp3 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=1000)


mlp1.fit(train_data, train_labels)
mlp2.fit(train_data, train_labels)
mlp3.fit(train_data, train_labels)


print("1 ukryta warstwa i 2 neurony")
predictions_train1 = mlp1.predict(train_data)
print(accuracy_score(predictions_train1, train_labels))
print()
print("1 ukryta warstwa i 3 neurony")
predictions_train2 = mlp2.predict(train_data)
print(accuracy_score(predictions_train2, train_labels))
print()
print("2 ukryte warstwy, 3 neurony")
predictions_train3 = mlp3.predict(train_data)
print(accuracy_score(predictions_train3, train_labels))