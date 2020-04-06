# -*- coding: utf-8 -*-

"""
・TLEだったやつを修正。これでもAC。
・DFS、全列挙、ソート
・1回全部列挙してソートだけすれば、にぶたんとか要らないじゃんね。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

MAX = 3234566667
ans = []
def rec(x):
    if int(x) > MAX:
        return
    ans.append(int(x))
    last = int(x[-1])
    rec(x+str(last))
    if last != 0:
        rec(x+str(last-1))
    if last != 9:
        rec(x+str(last+1))
    return

K = INT()

for i in range(1, 10):
    rec(str(i))

ans.sort()
print(ans[K-1])
