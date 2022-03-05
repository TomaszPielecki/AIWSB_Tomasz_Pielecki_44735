import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('miasta.csv')
print(df)

dziesiata = {
    'Rok' : ['2010'],
    'Gdansk' : ['460'],
    'Poznan' : ['555'],
    'Szczecin' : ['405']
}
df.append({'Rok':2010,'Gdansk':460,'Poznan':555,'Szczecin':405},ignore_index=True)
print(df)
plt.plot(df['Rok'],df['Gdansk'],color="red",marker="o")
plt.xlabel('Lata')
plt.ylabel('Liczba Ludnosci w tys')
plt.title('Ludnosc w miastach Polski')
plt.show()

plt.plot(df['Rok'],df['Gdansk'],color="red",marker="o")
plt.plot(df['Rok'],df['Poznan'],color="green",marker="o")
plt.plot(df['Rok'],df['Szczecin'],color="blue",marker="o")
plt.xlabel('Lata')
plt.ylabel('Liczba Ludnosci w tys')
plt.title('Ludnosc w miastach Polski')
plt.legend(['Gdańsk','Poznań','Szczecin'])
plt.show()