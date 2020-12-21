"""
・さくっと自力AC、1ペナしたけど。
・場合分け
・0と1しかないので、ちょっと長くなればすぐ被る。真面目な文字列探索はやらなくて大丈夫。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
S = input()

if N == 1:
    NO()
elif N == 2:
    if S[0] == S[1]:
        YES()
    else:
        NO()
elif N == 3:
    if S == '010' or S == '101':
        NO()
    else:
        YES()
else:
    YES()
