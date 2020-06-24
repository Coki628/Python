"""
参考：https://twitter.com/lorent_kyopro/status/1275460543977910272
　　　https://twitter.com/titia_til/status/1275460172647821315
　　　https://twitter.com/long_ng_cp/status/1275465525099782145
・自力ならず。。
・数え上げDP、前計算
・実験が足りなかった。面倒がらずに手元でもっと先まで書いて観察するべきだったな。
・7,8項目くらいまで書くと見えそう。左右対称なので半分だけ書くといくらか楽。(自分ノート参照)
・N項目は、根にN-2項目が2つ(左右)とN-1項目が1つ(真ん中)がくっついたような状態になる。
・ここから dp[i] = dp[i-1] + dp[i-2]*2 を考えることができて、このdp[i]をi項目で作れる爪の数とすると、
　3回に1回はN-2,N-1の状態とは別に根から4つ使ってもう1個作ることができる、となる。
　これを最大値までDPする、を前計算でやっておく。
・独立した複数テストケースだけど、今回はDPの内容が常に同じで前計算しておけるのでやる。
　地味にそこに気づかなかった可能性もあるな。クエリならやるけど、
　複数テストケースで前計算するの珍しいからな。
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

MAX = 2 * 10**6
# dp[i] := i項目で作れる爪の数
dp = [0] * (MAX+1)
dp[3] = 1
for i in range(4, MAX+1):
    dp[i] = dp[i-1] + dp[i-2]*2 + (i%3 == 0)
    dp[i] %= MOD

for _ in range(INT()):
    N = INT()

    ans = dp[N]*4 % MOD
    print(ans)
