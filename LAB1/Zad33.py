def moja_funkcja():
    print("Moja funkcja")
    print("cos wypisuje")

def dzialania(a, b):
    S = a + b
    I = a * b
    D = a / b
    return [S, I, D]

moja_funkcja()
moja_funkcja()

wynik = dzialania(10, 5)
print(wynik)