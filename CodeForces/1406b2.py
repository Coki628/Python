"""
参考：https://codeforces.com/contest/1406/submission/92591339
・ツイッターでのじゅぴろさんの話から。
・耳DP
・とりあえず自分で書いてみて、あ、マイナスからいきなり大きくなる時どう対処しよう、
　ってなったけど、実装見て納得。最小値も一緒に持てばいいんだ。
　そうしたら正負が変わった時に最大値になるやつも全部カバーできる。
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
INF = 10**19
MOD = 10**9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    M = 5
    dp0 = list2d(N+1, M+1, -INF)
    dp1 = list2d(N+1, M+1, INF)
    dp0[0][0] = dp1[0][0] = 1
    for i in range(N):
        for j in range(M+1):
            if dp0[i][j] != -INF:
                dp0[i+1][j] = max(dp0[i+1][j], dp0[i][j])
                if j < M:
                    dp0[i+1][j+1] = max(dp0[i+1][j+1], dp0[i][j]*A[i])
                    dp1[i+1][j+1] = min(dp1[i+1][j+1], dp0[i][j]*A[i])
            if dp1[i][j] != INF:
                dp1[i+1][j] = min(dp1[i+1][j], dp1[i][j])
                if j < M:
                    dp1[i+1][j+1] = min(dp1[i+1][j+1], dp1[i][j]*A[i])
                    dp0[i+1][j+1] = max(dp0[i+1][j+1], dp1[i][j]*A[i])
    ans = dp0[N][M]
    print(ans)
