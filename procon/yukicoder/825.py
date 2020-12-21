# -*- coding: utf-8 -*-

"""
・自力AC
・制約が小さいので、所持コイン2種類と購入額の3重ループを全探索。
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

a, b, c = MAP()

ans = INF
for i in range(a+1):
    for j in range(b+1):
        unused = (a-i) + (b-j)
        pay = i + j*10
        for g in range(1, pay+1):
            change = pay - g
            if change < 0:
                break
            cnt1, cnt10 = divmod(change, 10)
            cnt = unused + cnt1 + cnt10
            if cnt == c:
                ans = min(ans, g)
if ans == INF:
    print('Impossible')
else:
    print(ans)
