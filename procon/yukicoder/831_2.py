"""
・自力AC
・DP、漸化式
・実験して数列をN=8くらいまで出したあと、2回階差取ったら良さげな性質が出たので
　(-1,+3を交互にしてるだけ)、それ使って3本の漸化式でDPした。
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

N = INT()

if N == 1:
    print(0)
    exit()

dp0 = [0] * (N+1)
dp1 = [0] * (N+1)
dp2 = [0] * (N+1)
dp0[1] = 1
dp1[1] = 3
dp2[1] = 4
for i in range(2, N+1):
    dp0[i] = dp0[i-1] + dp1[i-1]
    dp1[i] = dp1[i-1] + dp2[i-1]
    if i % 2 == 0:
        dp2[i] = dp2[i-1] - 1
    else:
        dp2[i] = dp2[i-1] + 3
ans = dp0[N]
print(ans)
