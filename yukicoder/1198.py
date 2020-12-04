"""
・自力AC
・実験して察する。数学問、ほんとこういう解き方しかできない。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()

# se = set()
# for a in range(1, 100):
#     aa = a * a
#     for b in range(1, a):
#         bb = b * b
#         res = aa - bb
#         se.add(res)
# li = []
# for i in range(1, 200):
#     if i not in se:
#         li.append(i)

if N == 1 or N == 4 or N % 4 == 2:
    print(-1)
else:
    print(1)
