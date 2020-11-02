"""
・自分がやりかけてた方の方針。添字が逆。
・こっちだと遷移もNかかるので計算量が3乗になるけど、
　これも時間の範囲NからN*2にしたらちゃんと通った。。惜しかったじゃん。。。
"""

import sys

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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()

    A.sort()
    dp = list2d(N+1, N*2+1, INF)
    dp[0][0] = 0
    for i in range(N):
        for j in range(N*2+1):
            for k in range(j+1, N*2+1):
                # i個目の料理を時間kに取り出す
                dp[i+1][k] = min(dp[i+1][k], dp[i][j] + abs(A[i]-k))
    ans = min(dp[N])
    print(ans)
