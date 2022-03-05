x = 29
if x == 0 or x == 1:
    print("pierwsza")
else:
    for i in range(2, x, 1):
        if x % i == 0:
            print("nie jest pierwsza")
            break
        elif i == x - 1:
            print("pierwsza")