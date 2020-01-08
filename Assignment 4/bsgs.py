from math import sqrt, floor


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


# returns the modular multiplicative inverse of a number mod m
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception("inverse does not exist")
    else:
        return x % m


def bsgs(g: int, h: int, p: int):
    m = int(floor(sqrt(p - 1)))
    g_i = []
    for i in range(m):
        g_i.append((g ** i) % p)

    k = (modinv(g, p)) ** m % p
    a = 0

    for j in range(m + 2):
        temp = h * (k ** j) % p
        if temp in g_i:
            a = g_i.index(temp) + (m * j)
            break

    print(a)


bsgs(2, 30, 103)
