"""
参考：https://twitter.com/ngtkana/status/1285617508171632643
　　　https://codeforces.com/contest/1382/submission/87588388
・自力ならず。
・部分和DP
・必要な考察として、開始時点から見ていって、それまでの最大より小さい要素は1つ前と同じ方に入れるしかない。
　自分は終了状態から遡って復元的な方針で進んでしまったため、この時点でダメだったんだな。。
・で、上記が分かると、連続で取らなきゃいけない各区間長毎でひとまとめにして新たな数列を作り、
　その各区間長を組み合わせてちょうど長さNが作れるかどうか、という問題になる。
・N^2いける制約なので、これは部分和DPをやることで解ける。
・部分和DP、久しぶりにやったけど、こーいう使い方もできるんだなー。
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
    N = INT()
    P = LIST()

    A = []
    j = 0
    for i in range(1, N*2):
        if P[i] > P[j]:
            A.append(i-j)
            j = i
    A.append(N*2-j)

    M = len(A)
    dp = list2d(M+1, N+1, 0)
    dp[0][0] = 1
    for i, a in enumerate(A):
        for j in range(N+1):
            if dp[i][j]:
                dp[i+1][j] |= dp[i][j]
                if j+a <= N:
                    dp[i+1][j+a] |= dp[i][j]
    if dp[M][N]:
        YES()
    else:
        NO()
