"""
・自力TLE。。
・ゲームDP
・意外と状態数少ないんじゃないかと思ったら、全然そんなことなかった。。。
"""

import sys
from collections import defaultdict

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

def rec(i, cur):
    if memo[i][cur] != -1:
        return memo[i][cur]
    if i == N:
        if cur == 0:
            return 0
        else:
            return 1
    if S[i] == '0':
        res = 1
        res &= rec(i+1, cur)
        res &= rec(i+1, cur^A[i])
    else:
        res = 0
        res |= rec(i+1, cur)
        res |= rec(i+1, cur^A[i])
    memo[i][cur] = res
    return res

for _ in range(INT()):
    N = INT()
    A = LIST()
    S = input()

    memo = [defaultdict(lambda: -1) for i in range(N+1)]
    ans = rec(0, 0)
    print(ans)
