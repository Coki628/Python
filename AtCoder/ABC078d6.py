# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-9
・これはTLE。この後必死で山札の添字を1つ減らせないか色々考えた。
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

N, _, W = MAP()
A = [W] + LIST()

# memo[i][j] := 先手が最後にi枚目、後手が最後にj枚目を取った状態
memo = list2d(N+1, N+1, None)
def rec(a, b):
    if memo[a][b] is not None:
        return memo[a][b]
    # 終了条件(どちらかが最後まで札を取った)
    if a == N or b == N:
        res = abs(A[a]-A[b])
        memo[a][b] = res
        return res
    nxt = max(a, b)
    turn = 0 if a < b else 1
    if turn % 2 == 0:
        res = -INF
        for i in range(nxt+1, N+1):
            res = max(res, rec(i, b))
    else:
        res = INF
        for i in range(nxt+1, N+1):
            res = min(res, rec(a, i))
    memo[a][b] = res
    return res

ans = rec(-1, 0)
print(ans)
