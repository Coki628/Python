"""
・なんとか自力ACー。大変だった。。
・周期
・最初と最後は愚直シミュレーション、真ん中は計算でO(1)する。
・前後どのくらい愚直やれば当たりが余ったりの部分をカバーできるか、
　正当性は証明できてないけど、最初の10万回くらいと最後の1周期+余りでやったらOKだった。
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

N, K = MAP()
A = list(map(int, input()))
# 10万回くらいで周期に合うところに合わせる
M = 10**5 // N * N

# とりあえず10万回くらいやる
cost = bonus = 0
for i in range(min(K, M)):
    if bonus == 0:
        cost += 1
    else:
        bonus -= 1
    if A[i%N] == 1:
        bonus += 1
    elif A[i%N] == 2:
        bonus += 2

# 当たりの数のが多いならある程度先からはコスト0でカバーできてるはず
if N <= sum(A):
    print(cost)
    exit()

# Kが大きい時
if K > M:
    K -= M
    # 最後の1周期+余りだけ愚直にやるように調整
    d, m = divmod(K, N)
    if d:
        d -= 1
        m += N
    # 間は計算で
    cost += N*d - sum(A)*d

    for i in range(m):
        if bonus == 0:
            cost += 1
        else:
            bonus -= 1
        if A[i%N] == 1:
            bonus += 1
        elif A[i%N] == 2:
            bonus += 2
print(cost)
