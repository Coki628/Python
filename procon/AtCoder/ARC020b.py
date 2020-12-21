# -*- coding: utf-8 -*-

"""
・自力AC
・N=100と制約が小さいので、10C2*2=90の全組み合わせの内側にNのループ回しても全く問題ない。
"""

import sys
from itertools import combinations

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

N, C = MAP()
A = [INT() for i in range(N)]

mn = INF
# 10色から2色選ぶ組み合わせ全部
for comb in combinations(range(1, 11), 2):
    for i in range(2):
        # 偶奇どっちにどっちを塗るか
        c1, c2 = comb[i], comb[1-i]
        cnt = 0
        # 完成形と違う色なら塗り替えコストを足す
        for j, a in enumerate(A):
            if j%2 == 0:
                if a != c1:
                    cnt += C
            else:
                if a != c2:
                    cnt += C
        # 全パターンから最小を取る
        mn = min(mn, cnt)
print(mn)
