# -*- coding: utf-8 -*-

# 組み合わせの数だけ返してくれる関数
def nCr(n, r):
    """
    Calculate the number of combination (nCr = nPr/r!).
    The parameters need to meet the condition of n >= r >= 0.
    It returns 1 if r == 0, which means there is one pattern
    to choice 0 items out of the number of n.
    """

    # 10C7 = 10C3
    r = min(r, n-r)

    # Calculate the numerator.
    numerator = 1
    for i in range(n, n-r, -1):
        numerator *= i

    # Calculate the denominator. Should use math.factorial?
    denominator = 1
    for i in range(r, 1, -1):
        denominator *= i

    return numerator // denominator

H,W,A,B = map(int, input().split())

h = H - A
w = B + 1
ans = 0
# マスを右上に1つずつずらして、必ず通る場所でパターンを足し合わせていく
while h > 0 and w <= W:
    ans += nCr(h+w-2, h-1) * nCr(H-h+W-w, H-h) % (10 ** 9 + 7)
    h -= 1
    w += 1

print(ans % (10 ** 9 + 7))