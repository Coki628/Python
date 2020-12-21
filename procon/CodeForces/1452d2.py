"""
・なんとか自力AC！
・1次元DP、高速化
・確率を聞かれてるけど、本質は数え上げDP。条件を満たすように電波塔を設置するには、
　それぞれに奇数長の区間をカバーさせればいいので、自分の1,3,5...個前からは遷移してこれる。
・最初、1個飛ばし累積和をやろうとしたらバグりまくって終了。。
　log付くけどBIT2本に偶数位置と奇数位置を管理させて区間和取得、でやったらすぐうまくいった。。
・ABC183Eの実装を参考にやったら累積和でも通った。最近やったのは覚えてたんだから、
　すぐ調べに行けばよかったな。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 998244353
EPS = 10**-10

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

N = INT()

dp = [0] * (N+1)
acc = [0] * (N+1)
acc[0] = acc[1] = 1
for i in range(1, N+1):
    dp[i] += acc[i-1]
    dp[i] %= MOD
    if i-2 >= 0:
        acc[i] = dp[i] + acc[i-2]
        acc[i] %= MOD
cnt = dp[N]

ans = fermat(cnt, pow(2, N, MOD), MOD)
print(ans)
