# -*- coding: utf-8 -*-

"""
・400点自力AC
・最初、累積和の復元的なことをした。
・で、意外と単純なやつでいけそうだったので、
　最大値10**9だけ別処理して、他は区間和が該当しないように10**9で埋めたりしてAC。
・公式解説見たけど、ちょっと問題勘違いしてた。
　構築する数列、広義単調増加じゃないといけない気がしてた(1<=l<=r<=Nってとこから)けどそんな制限なくて何でも良かった。
　ならもっと簡単に作れたのにな。。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, K, S = MAP()

MAX = 10 ** 9
if S == MAX:
    A = [1] * (N-K) + [MAX] * K
    print(*A)
    exit()

A = [S] * K + [MAX] * (N-K)
print(*A)
