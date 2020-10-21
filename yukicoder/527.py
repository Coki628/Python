"""
・自力AC！
・ちょこっとどうしようか迷ったけど、100*10万=1000万は普通にDPで間に合うな、となったらやるだけ。
・配列は節約しようか迷ったけど、とりあえずそのまま行った。(コンテスト中なら念のため節約するかも)
・結果的には、pypyAC0.28秒、メモリが150MBで問題なし。(512MB制限なのでまあ余裕)
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

N = INT()
VW = []
for i in range(N):
    v, w = MAP()
    VW.append((v, w))
V = INT()

MX = N*1000
dp = list2d(N+1, MX+1, 0)
dp[0][0] = 0
for i, (v, w) in enumerate(VW):
    for j in range(MX+1):
        dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        if j+w <= MX:
            dp[i+1][j+w] = max(dp[i+1][j+w], dp[i][j] + v)

ans = [-1] * 2
for j in range(1, MX+1):
    if dp[N][j] == V:
        if ans[0] == -1:
            ans[0] = j
        ans[1] = j
print(ans[0])
if ans[1] == MX:
    print('inf')
else:
    print(ans[1])
