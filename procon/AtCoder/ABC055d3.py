# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・証言から確実なところを埋めていって矛盾なく終われるか確認する系
・今回は最初の2つを決め打つと、後が定まる。
"""

import sys
from itertools import product

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
S = input()
S += S[0]

def check(a, b):
    res = [a, b]
    # 自分と1つ前がS,Wどっちかと、自分の発言のo,xが分かれば、次が定まる
    for i in range(1, N+1):
        # 隣が同じで発言がo,違ってxなら羊
        if res[i-1] == res[i] and S[i] == 'o' \
                or res[i-1] != res[i] and S[i] == 'x':
            res.append('S')
        # 隣が同じで発言がx,違ってoなら狼
        else:
            res.append('W')
    # 1周して矛盾していないか
    if res[0] == res[N] and res[1] == res[N+1]:
        return res
    else:
        return []

# 1,2番目がS,Wどっちかの4通り試す
for a, b in product(['S', 'W'], repeat=2):
    res = check(a, b)
    if res:
        print(''.join(res[:N]))
        exit()
print(-1)
