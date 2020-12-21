"""
・0未満の各位置について、初めてそこに達するindexを持っておく。
・これで途中でNGになるパターンの操作回数を網羅できるので、後はOK時のN回を足せば答え。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    S = input()
    N = len(S)

    D = {}
    cur = 0
    for i, s in enumerate(S):
        if s == '+':
            cur += 1
        else:
            cur -= 1
            if cur < 0 and cur not in D:
                D[cur] = i + 1
    ans = sum(D.values()) + N
    print(ans)
