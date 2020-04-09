# -*- coding: utf-8 -*-

"""
・なんとか自力AC！
・累積和応用
・累積和で値が同値になるペアを見れば区間和が0になる場所は分かる。
・これを使って、ある左端から一番最初に同値が見つかる場所を取っておく。
　この手前まで右端として使えることになる。
・ただし、自分より先で、自分より早く右端が終わるところがあれば、
　自分もその場所より先には行けないので、後ろから累積minしておく。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

acc = [0] + list(accumulate(A))
D = {}
# R[i] := 左端をiとした時、右端として使える最大位置
R = [N] * N
for i, a in enumerate(acc):
    if a in D:
        # 同値が見つかったら、その1つ手前まで
        R[D[a]] = i - 1
    D[a] = i
# 後ろから累積min
R = list(accumulate(R[::-1], min))[::-1]
ans = 0
for i in range(N):
    ans += R[i] - i
print(ans)
