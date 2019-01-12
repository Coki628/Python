# -*- coding: utf-8 -*-

"""
・前保留にしてたやつやった。
・循環しているのをmodでうまく処理する。
"""

n, m = map(int, input().split())

cnt = 0
# nのが少ない時は高々1000以下なので愚直にできる
if n <= m:
    for i in range(1, n+1):
        for j in range(1, n+1):
            if (pow(i, 2) + pow(j, 2)) % m == 0:
                cnt += 1
# nが大きい時
else:
    # まずmまでの範囲にあるやつを数える
    for i in range(1, m+1):
        for j in range(1, m+1):
            if (pow(i, 2) + pow(j, 2)) % m == 0:
                cnt += 1
    # 循環しているはずなので、割り切れる所まで追加
    cnt *= pow(n // m, 2)
    # 右と下に余る分(今回は例えば6,2も2,6も一緒なので、最後に*2する)
    a = 0
    for i in range(1, m+1):
        for j in range(1, n%m+1):
            if (pow(i, 2) + pow(j, 2)) % m == 0:
                a += 1
    # 余り切り捨てがあるので、n//mは括弧つけて先にやる
    cnt += a * (n // m) * 2
    # 右下に余る分
    for i in range(1, n%m+1):
        for j in range(1, n%m+1):
            if (pow(i, 2) + pow(j, 2)) % m == 0:
                cnt += 1
print(cnt)
