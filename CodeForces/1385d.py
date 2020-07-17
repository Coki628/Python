"""
・自力AC！
・再帰探索
・2冪を半分ずつにしていく感じで、常にいい方の手を採用していけば最適にできる。
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

def rec(i, ln, c):
    if ln == 1:
        return S[i] == c
    res = max(
        # この区間の後半を文字cにする
        rec(i, ln//2, c+1) + S[i+ln//2:i+ln].count(c),
        # この区間の前半を文字cにする
        rec(i+ln//2, ln//2, c+1) + S[i:i+ln//2].count(c),
    )
    return res

for _ in range(INT()):
    N = INT()
    S = [ord(c)-97 for c in input()]

    res = rec(0, N, 0)
    ans = N - res
    print(ans)
