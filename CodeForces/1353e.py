"""
参考：https://codeforces.com/contest/1353/submission/80101597
・自力ならず。累積和とかでやろうとしてたけどうまくいかず終了。。
・耳DP
・MOD毎に見てDPする。2重ループでもO(N)で済む。
・状態は、ランプをつけ始める前、つける区間、つけ終わった後、の3つ。
・添字が飛び飛びになるので、DP配列は直前だけ持って、パリティで管理した。
・pypyAC0.6秒。(制約1秒)
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

    cnt1all = S.count(1)
    ans = INF
    for i in range(K):
        dp0 = [INF] * 2
        dp1 = [INF] * 2
        dp2 = [INF] * 2
        dp0[0] = 0
        cnt1 = 0
        j = i
        p = 0
        while j < N:
            cnt1 += (S[j] == 1)
            dp0[1-p] = dp1[1-p] = dp2[1-p] = INF
            dp0[1-p] = min(dp0[1-p], dp0[p] + (S[j] == 1))
            dp1[1-p] = min(dp1[1-p], dp0[p] + (S[j] == 0))
            dp1[1-p] = min(dp1[1-p], dp1[p] + (S[j] == 0))
            dp2[1-p] = min(dp2[1-p], dp1[p] + (S[j] == 1))
            dp2[1-p] = min(dp2[1-p], dp2[p] + (S[j] == 1))
            j += K
            p = 1 - p
        cost = min(dp0[p], dp1[p], dp2[p])
        ans = min(ans, cnt1all - cnt1 + cost)
    print(ans)
