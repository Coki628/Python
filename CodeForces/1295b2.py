# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/kiri8128/status/1222561158403514369
　　　https://codeforces.com/contest/1295/submission/69744899
・とりあえず半写経でACしたけどモヤモヤ。。
・1周した値でmodとればうまく1回にまとめられるのは分かったけど、
　これどうすればこうなるって分かるんだろうな。。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):

    N, X = MAP()
    S = input()

    acc = [1 if c == '0' else -1 for c in S]
    acc = [0] + list(accumulate(acc))
    mod = acc[-1]
    acc = acc[:-1]
    if mod > 0:
        cnt = 0
        for a in acc:
            if (X-a) % mod == 0 and X-a >= 0:
                cnt += 1
        print(cnt)
    elif mod < 0:
        cnt = 0
        for a in acc:
            if (X-a) % mod == 0 and X-a <= 0:
                cnt += 1
        print(cnt)
    else:
        for a in acc:
            if a == X:
                print(-1)
                break
        else:
            print(0)
