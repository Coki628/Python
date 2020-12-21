"""
・自力AC！
・区間スケジューリング、耳DP、貰うDP
・題意を満たすことを考えると、重複しない区間をなるべく多く取る(接するのはOK)、
　ただし区間1つ取るのにもコストがかかるので、その区間を取った場合の利益と足し引きする、
　をうまくやれるといい。に帰着する。
・区間を効率よく取るには、区間スケジューリングっぽく右端ソートした各要素を順番に見ていく、をやる。
・今回は0とNが絡む場合だけ取得コストが変わるので場合分けする。
・ある区間を取りたい時、既に左端に線が引いてあるかどうかを耳として持ってDP。
・だいたいの方針はすぐ立ってたけど1ペナ。
　結局、dp1の初期値が0だったので、線が引いてない場所からでも、左端に線が引いてあるものとして
　遷移できてしまっていたのが原因。やっぱ初期値は常にINF,-INFが無難か。。
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
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N, M, K = MAP()

RL = [[] for i in range(N+1)]
for i in range(M):
    l, r, p = MAP()
    l -= 1
    RL[r].append((l, p))

# dp0[i] := 位置iまでで取れる得点の最大値(位置iに線を引いていない)
dp0 = [-INF] * (N+1)
# dp1[i] := 位置iまでで取れる得点の最大値(位置iに線を引いた)
dp1 = [-INF] * (N+1)
dp0[0] = dp1[0] = 0
for r in range(1, N+1):
    # 位置rに線を引かない遷移
    dp0[r] = max(dp0[r-1], dp1[r-1])
    # 位置rに線を引く遷移
    for l, p in RL[r]:
        # 右端がNの時は右端に線を引くコストがかからないので場合分け
        if r == N:
            dp1[r] = max(dp1[r], dp0[l] + p-K)
            dp1[r] = max(dp1[r], dp1[l] + p)
        else:
            dp1[r] = max(dp1[r], dp0[l] + p-K*2)
            dp1[r] = max(dp1[r], dp1[l] + p-K)
ans = max(dp0[N], dp1[N])
print(ans)
