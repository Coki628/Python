"""
参考：https://img.atcoder.jp/agc046/editorial.pdf
　　　https://atcoder.jp/contests/agc046/submissions/14504388
・こっちのが公式解な感じ。
・数え上げDP
・最後に増やしたのが行or列で耳っぽくDPを分けて、向きが切り替わる時には片側しか増やさないようにする。
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
INF = 10 ** 19
MOD = 998244353

a, b, c, d = MAP()
dp0 = list2d(c+1, d+1, 0)
dp1 = list2d(c+1, d+1, 0)

dp1[a][b] = 1
for i in range(a, c+1):
    for j in range(b, d+1):
        # 行を増やす遷移
        dp0[i][j] += dp0[i-1][j] * j
        dp0[i][j] += dp1[i-1][j] * j
        # 列を増やす遷移
        dp1[i][j] += dp1[i][j-1] * i
        # 向きが切り替わる時には片側しか増やさないようにする
        dp1[i][j] += dp0[i][j-1]
        dp0[i][j] %= MOD
        dp1[i][j] %= MOD
ans = (dp0[c][d] + dp1[c][d]) % MOD
print(ans)
