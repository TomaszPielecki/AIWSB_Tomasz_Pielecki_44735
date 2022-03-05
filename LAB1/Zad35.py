import numpy
import numpy as np

x = np.array([3, 8, 9, 10, 12])
y = np.array([8, 7, 7, 5, 6])

print("suma: ", x + y)
print("iloczyn :", x * y)

print("iloczyn skalarny: ", np.dot(x, y))
print("długość eukliesowa: ", numpy.linalg.norm(x))

z = np.random.randint(1, 100, 50, int)
print("randomowy zakres: ",z)
print("Średnia: ", numpy.average(z))
print("minimum to:", numpy.min(z), ", max to: ", np.max(z))
print("odchylenie standardowe: ", np.std(z))

normalized_z = (z - np.min(z)) / (np.max(z) - np.min(z))
print("wektor znormalizowany: ",normalized_z)