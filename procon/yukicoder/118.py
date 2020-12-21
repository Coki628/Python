"""
・なんとか自力AC
・数え上げ
・3つの要素が異なる数を数え上げる。最小値を決め打ちして、そこより大きい値から2つ取る。
　それぞれの値は数が異なるので、個別に掛けたものを合計に足していく。
・要素の取りうる値は1〜100までなので、個別に足していっても十分間に合う。
"""

import sys
from collections import Counter
from itertools import combinations

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
A = LIST()

C = sorted(Counter(A).items())
M = len(C)
ans = 0
for i, (k1, v1) in enumerate(C):
    for comb in combinations(C[i+1:], 2):
        k2, v2 = comb[0]
        k3, v3 = comb[1]
        ans += v1 * v2 * v3
        ans %= MOD
print(ans)
