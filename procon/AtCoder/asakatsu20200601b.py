"""
・正規表現
・前やった時そこそこ時間かかったぽいから、今回すんなり通せてよかった。
"""

import sys
import re

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

S = 'keyence'
N = len(S)
T = input()
M = len(T)

for i in range(N+1):
    S1 = S[:i]
    S2 = S[i:]
    exp = '^{0}(.*){1}$'.format(S1, S2)
    if re.match(exp, T):
        YES()
        exit()
NO()
