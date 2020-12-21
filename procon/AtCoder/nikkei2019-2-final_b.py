# -*- coding: utf-8 -*-

"""
・自力AC！
・文字列検索、スライス普通に毎回やったら重いかと思ったけど、意外と通った。
・N<=500に3重ループしてるから全部だと1億超だけど、実際はそんなに回ってない。
・pypyAC0.3秒、pythonでもAC1.9秒。
"""

import sys

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

S = input()
N = len(S)

cnt = 0
for i in range(2, N-3):
    for j in range(i+2, N-1):
        # 3分割する
        S1 = S[:i]
        S2 = S[i:j]
        S3 = S[j:]
        # 真ん中の条件
        if len(S2) % 2 == 1 or S2[len(S2)//2:] != S2[:len(S2)//2]:
            continue
        # 前後の条件
        m = min(len(S1), len(S3))
        k = 1
        while k < m:
            # suffixが合う限り条件を満たす
            if S1[-k] == S3[-k]:
                cnt += 1
            else:
                break
            k += 1
print(cnt)
