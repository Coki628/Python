# -*- coding: utf-8 -*-

"""
・共通処理関数化してスッキリ
"""

# 何回も使うから関数化しといた
def calc(x, y):
    cnt = 0
    for i in range(1, y+1):
        for j in range(1, x+1):
            if (pow(i, 2) + pow(j, 2)) % m == 0:
                cnt += 1
    return cnt

n, m = map(int, input().split())

cnt = 0
# nのが少ない時は高々1000以下なので愚直にできる
if n <= m:
    cnt = calc(n ,n)
# nが大きい時
else:
    # まずmまでの範囲にあるやつを数える
    cnt = calc(m ,m)
    # 循環しているはずなので、割り切れる所まで追加
    cnt *= pow(n // m, 2)
    # 右と下に余る分(今回は例えば6,2も2,6も一緒なので、最後に*2する)
    a = calc(n%m, m)
    # 余り切り捨てがあるので、n//mは括弧つけて先にやる
    cnt += a * (n // m) * 2
    # 右下に余る分
    cnt += calc(n%m, n%m)
print(cnt)
