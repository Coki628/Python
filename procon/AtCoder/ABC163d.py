# -*- coding: utf-8 -*-

"""
・終了8分前頃にギリギリAC。遅すぎた。
・数え上げ、累積和
・nCrから重複分を引けないかとかの方針で迷走した。重複大量にありすぎた。
・結局、最小から最大までの間は全て網羅できるので1つずつ、に辿り着いたのが終了間際になってから。
・考察力、、どうすればもっと養えるもんかね。。
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

N, K = MAP()

acc = list(accumulate(range(N+2)))
ans = 0
for i in range(K, N+2):
    # i個で作れる和の最大 - 最小 + 1
    ans += (acc[N+1] - acc[N+1-i]) - acc[i] + 1
    ans %= MOD
print(ans)
