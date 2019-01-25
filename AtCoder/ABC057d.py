# -*- coding: utf-8 -*-

"""
・自力AC
・割り算結果はint()だと.99とか切り捨てられるからround()する。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from collections import Counter
from math import factorial

def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) / factorial(r) / factorial(n-r)

round = lambda x: int((x*2+1) // 2)

N, A, B = map(int, input().split())
vN = list(map(int, input().split()))

# 組み合わせ計算用に、各値の個数も出しておく
c = Counter(vN)
# 大きい値から欲しいので降順ソートしておく
vN.sort(reverse=True)

mx_avg = sum(vN[:A]) / A
lists = [vN[:A]]
for i in range(A+1, B+1):
    avg = sum(vN[:i]) / i
    if avg == mx_avg:
        # 平均値が最大になるなら、選び方リストに追加
        lists.append(vN[:i])

ans = 0
for l in lists:
    # 選び方を、同じ値の個数毎にまとめる
    c2 = Counter(l)
    combs = 1
    for k, v in c2.items():
        # 同じ値が複数ある場合の組み合わせを出す
        combs *= nCr(c[k], v)
    ans += combs
print(mx_avg)
print(round(ans))
