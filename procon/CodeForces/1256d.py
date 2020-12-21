# -*- coding: utf-8 -*-

"""
・結構すんなり自力AC
・辞書順最小
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()
    S = input()

    # 最大でどのくらい前まで持ってきたいかを前計算しておく
    ST = []
    i = 0
    for j in range(N):
        if S[j] == '0':
            ST.append((i, j))
            i += 1
    
    ans = ['1'] * N
    cnt = K
    for s, t in ST:
        dist = t - s
        # 残り操作回数が足りる限り一番前へ
        if dist <= cnt:
            cnt -= dist
            ans[s] = '0'
        else:
            # 足りなくなったら最後は帳尻合う分だけ前へ
            ans[s+dist-cnt] = '0'
            cnt = 0
    print(''.join(ans))
