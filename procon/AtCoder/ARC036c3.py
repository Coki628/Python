# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc036
　　　https://kenkoooo.hatenablog.com/entry/2015/04/06/230854
・桁DP
・区間最大を考える時、今のi位置より手前が右端のケースは手前からの遷移の時に織り込まれているので、
　自分の位置iを右端とした区間だけを常に考えればOK。
・で、その時点での最大がどう変化するかを考えると、これは+1と-1(0は0のまま)をすればOK。
　今回使う数を+1するのは自然だとして、それ以外は1つ桁が進む分、左に離れていくので、
　右端を固定してる都合上、絶対-1される感じ。
・普通に提出したら最大っぽいケース1つだけTLE。。
・1度遷移が始まってから0になったらもうその先は0、を使って枝刈り。
・最初WAで、？ってなったけど、多分MOD取った時にたまたま途中が0になっちゃう場合があるんだと思う。
・0が2回連続したら、にしたら無事速度的には間に合うようになって、今度はMLE。
・定番の配列2つだけにするをやって無事AC。pypyAC1.6秒。
・計算量は300^3=2700万。
　pypyなら通りそうな量だけど、前も3重ループで均等に300のこの形は確か枝刈りやんないと無理だった。(edpc_j)
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
S = input()

# dp[i][j][k] := i桁目まで見て、今の桁を右端とした任意の区間で、0の最大がj、1の最大がkの時の通り数
dp = list3d(2, K+1, K+1, 0)
dp[0][0][0] = 1
for i, s in enumerate(S):
    cur = i % 2
    nxt = (i+1) % 2
    s = S[i]
    for j in range(K+1):
        started = False
        for k in range(K+1):
            if dp[cur][j][k] > 0:
                started = True
            # 枝刈り
            if started and k < K and dp[cur][j][k] == dp[cur][j][k+1] == 0:
                break
            if s == '?':
                # 今回の桁を0にする
                if j+1 <= K:
                    dp[nxt][j+1][max(0, k-1)] += dp[cur][j][k]
                    dp[nxt][j+1][max(0, k-1)] %= MOD
                # 今回の桁を1にする
                if k+1 <= K:
                    dp[nxt][max(0, j-1)][k+1] += dp[cur][j][k]
                    dp[nxt][max(0, j-1)][k+1] %= MOD
            elif s == '0':
                if j+1 <= K:
                    dp[nxt][j+1][max(0, k-1)] += dp[cur][j][k]
                    dp[nxt][j+1][max(0, k-1)] %= MOD
            else:
                if k+1 <= K:
                    dp[nxt][max(0, j-1)][k+1] += dp[cur][j][k]
                    dp[nxt][max(0, j-1)][k+1] %= MOD
    dp[cur] = list2d(K+1, K+1, 0)
ans = 0
for j in range(K+1):
    for k in range(K+1):
        ans += dp[N%2][j][k]
        ans %= MOD
print(ans)
