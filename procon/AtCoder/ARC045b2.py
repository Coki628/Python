# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc045
・公式解
・セグ木は使わない。
・大丈夫な区間(2以上の場所)は0、ダメな区間(1の場所)は1にする、いわゆる2値化する。
・で、そいつで累積和もう1回取ってその区間和から、ダメな区間を含むか判定する。
・python、pypyともに0.4秒AC。やっぱセグ木より累積和のが全然速いや。
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
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
ST = []
imos = [0] * (N+2)
for i in range(M):
    s, t = MAP()
    ST.append((s, t))
    imos[s] += 1
    imos[t+1] -= 1
# いもすで区間の重複を集計
imos = list(accumulate(imos))

is_one = [0] * (N+2)
for i in range(N+2):
    if imos[i] == 1:
        is_one[i] = 1
# 1つしかない場所を1、複数ある場所を0として、区間チェックできるように累積和を取る
is_one = list(accumulate(is_one))

ans = []
for i in range(M):
    # 区間[s,t]に値が1になる場所がない(全0)ならOK
    s, t = ST[i]
    # 累積和を取ってあるので、その区間が全0かどうかはすぐ判定できる
    if is_one[t] - is_one[s-1] == 0:
        ans.append(i+1)

print(len(ans))
for i in ans:
    print(i)
