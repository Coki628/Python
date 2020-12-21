# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/tyawanmusi512/status/1218889632437260289
・久しぶりにABC-D400点自力ならず。。
　いくら悩んでもゴミみたいな解法しか出なかった。。。
・N以下の数全てを最初の桁と最後の桁毎に集計
・後は例えば1***3は3***1な数全てとペアになりうるので、掛け合わせるだけ。
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

N = INT()

C = list2d(10, 10, 0)
for a in range(1, N+1):
    # 最初の桁と最後の桁毎に集計
    s = int(str(a)[0])
    t = int(str(a)[-1])
    C[s][t] += 1

ans = 0
for i in range(1, 10):
    for j in range(1, 10):
        ans += C[i][j] * C[j][i]
print(ans)
