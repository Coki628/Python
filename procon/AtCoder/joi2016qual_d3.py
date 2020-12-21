# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2016/2017-yo/2017-yo-t4/review/2017-yo-t4-review.html
・事前に2値化した累積和を各Mについて準備することで、内ループのN(10万)が削れる。
・がしかしこれでも部分点2で状況は変わらず。
"""

import sys
import numpy as np
from itertools import permutations
from collections import Counter

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
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
A = [a-1 for a in LIST(N)]

# 現在位置の区間和をO(1)取得するために累積和しておく
acc = np.zeros((M, N+1), dtype=np.int64)
for i, a in enumerate(A):
    # 区間にある個数が欲しいので、あるorないで2値化
    acc[a,i+1] = 1
for i in range(M):
    acc[i] = acc[i].cumsum()

C = Counter(A)
ans = INF
for perm in permutations(range(M)):
    # 完成形の持ち方もAと同じ形からランレングスに変更
    good = []
    idx = 0
    for a in perm:
        # (種類, 開始index, 終了index)
        good.append((a, idx, idx+C[a]))
        idx += C[a]

    diff = 0
    for a, s, t in good:
        # 移動が必要な個数 = aの総数 - 区間に含まれる個数(動かさなくていい個数)
        diff += C[a] - (acc[a,t]-acc[a,s])
    ans = min(ans, diff)
print(ans)
