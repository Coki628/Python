# -*- coding: utf-8 -*-

"""
・なんとか自力AC。考察、時間かかり過ぎたけど。。
・多分、Dが早めに通せてたとしても、間に合わなかったと思う。考察、爆速で整うようになればいいのに。
・数え上げ系。考え方はソース内コメント参照。
・この方針だと、ありえないのが来たら勝手に*0されて問答無用で通り数0になるのでそこも具合がいい。
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
A = LIST()

# C[i] := 今現在、次にiが来るのを待っている色の数
C = [0] * (N+1)
# 最初は3色とも0待ち
C[0] = 3

ans = 1
for a in A:
    # 同じ数を待っている色の数だけ、この位置での通り数がある
    ans *= C[a]
    ans %= MOD
    # 1色は待ちが次の数になるので、それに合わせて足し引きする
    C[a] -= 1
    C[a+1] += 1
print(ans)
