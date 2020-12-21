"""
参考：https://codeforces.com/contest/1353/submission/80147686
・自力ならず。累積和とかでやろうとしてたけどうまくいかず終了。。
・耳DP
・MOD毎に見てDPする。2重ループでもO(N)で済む。
・状態は、ランプをつけ始める前、つける区間、つけ終わった後、の3つ。
・予めMOD K毎で値を振り分けておいて、その各リストの長さを元に内ループでDPする。
　こっちのが添え字飛び飛びでループさせるより馴染みがあるかもしれない。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()
    S = list(map(int, input()))

    A = [[] for i in range(K)]
    for i, s in enumerate(S):
        A[i%K].append(s)

    cnt1all = S.count(1)
    ans = INF
    for i in range(K):
        M = len(A[i])
        dp0 = [INF] * (M+1)
        dp1 = [INF] * (M+1)
        dp2 = [INF] * (M+1)
        dp0[0] = 0
        cnt1 = 0
        for j in range(M):
            cnt1 += (A[i][j] == 1)
            dp0[j+1] = min(dp0[j+1], dp0[j] + (A[i][j] == 1))
            dp1[j+1] = min(dp1[j+1], dp0[j] + (A[i][j] == 0))
            dp1[j+1] = min(dp1[j+1], dp1[j] + (A[i][j] == 0))
            dp2[j+1] = min(dp2[j+1], dp1[j] + (A[i][j] == 1))
            dp2[j+1] = min(dp2[j+1], dp2[j] + (A[i][j] == 1))
        cost = min(dp0[M], dp1[M], dp2[M])
        ans = min(ans, cnt1all - cnt1 + cost)
    print(ans)
