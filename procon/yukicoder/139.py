"""
・自力AC！
・周期頑張ってシミュレーション
・それぞれの信号はT*2周期で動いているので、現在時間と到着時間でmod取って、
　両方周期の前半に収まってればそのまま渡れるし、そうでないなら待つ。
・境界の不等号の調整ミスで1ペナ。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, L = MAP()
X, W, T = [0] * (N+1), [0] * (N+1), [0] * (N+1)
for i in range(1, N+1):
    X[i], W[i], T[i] = MAP()

cur = 0
for i in range(1, N+1):
    # 直前の信号からここまでの移動
    cur += X[i] - (X[i-1]+W[i-1])
    # 渡りきる予定時間
    nxt = cur + W[i]
    # 周期のどこに当たるか
    mod = T[i]*2
    fr = cur % mod
    to = nxt % mod
    # どちらかが後半にかかっていたら待つ(出発の方は境界含む)
    if fr >= T[i] or to > T[i]:
        cur += (mod-fr) + W[i]
    else:
        cur = nxt
# 最後の目的地までの移動
cur += L - (X[N]+W[N])
print(cur)
