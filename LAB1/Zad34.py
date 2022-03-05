def obcinaj_s(s):
    wynik = s[1:-1]
    return wynik


def max_sum(a, b, c):
    if a >= c and b >= c:
        suma = a + b
    elif a >= b and c >= b:
        suma = a + c
    else:
        suma = b + c
    return suma


def wybierz_parzyste(x):
    zbior = []
    for i in range(1, len(x) + 1, 1):
        if i % 2 == 0:
            zbior.append(i)
    return zbior


def pole_trapezu(a, b, h):
    pole = ((a + b) * h) / 2
    return pole


print("obcinaj: ",obcinaj_s('wyraz'))
print("max suma: ", max_sum(5, 6, 7))
print("Pole trapezu: ", pole_trapezu(2, 5, 3))
print("Parzyste: ", wybierz_parzyste([1, 2, 3, 4, 5, 6, 7, 8]))