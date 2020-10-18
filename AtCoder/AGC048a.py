"""
・なんとか自力AC
・苦しい戦いだった。。
・色々ゴチャゴチャ書いてダメだったあと、最初から辞書順が大きいやつをカバーしきれてない気がして、
　その例外処理入れたらAC。。
"""

import sys
from string import ascii_lowercase

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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

for _ in range(INT()):
    S = input()
    N = len(S)

    ans = INF
    T = ascii_lowercase[1:]
    U = list('atcoder')
    M = len(U)
    if list(S) > U:
        print(0)
        continue
    for i in range(N):
        if S[i] in T:
            ans = min(ans, i)
            j = i - 1
            while j >= 0:
                if j < M:
                    S2 = list(S[:M])
                    S2.insert(j, S[i])
                    S2 = S2[:M]
                    if S2 > U:
                        ans = min(ans, i - j)
                        break
                j -= 1
    if ans == INF:
        print(-1)
    else:
        print(ans)
