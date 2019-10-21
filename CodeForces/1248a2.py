# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1248/submission/62984907
・最初は交点数だと思ったんだ。でもその後、問題文読み違えて列数だと思ってしまった。
・普通に交点数なら解けてたはず。けど、読み違えとはいえ、
　4500人が通したAを通せなかったのはつらい。。
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

for i in range(INT()):
    C = [0] * 2
    ans = 0
    N = INT()
    for a in LIST():
        # 偶奇に分けて集計
        C[a%2] += 1
    M = INT()
    for b in LIST():
        # 偶数は偶数、奇数は奇数の数だけ、整数座標に交点を作る
        ans += C[b%2]
    print(ans)
