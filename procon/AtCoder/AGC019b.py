# -*- coding: utf-8 -*-

"""
・500点自力AC！
・始点終点が同じ文字なら、反転させても前後1つずつ短くしたやつと被る。
・ほぼ想定解の考え方で進められてよかった。
"""

import sys
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

S = input()
N = len(S)

C = Counter(S)
# 区間(の始点終点ペア)の全組み合わせ + 元の文字列
ans = N*(N-1) // 2 + 1
for k, v in C.items():
    # 重複分を引いていく
    ans -= v*(v-1) // 2
print(ans)
