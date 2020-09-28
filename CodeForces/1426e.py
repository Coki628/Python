"""
・F先にやってこれを最後に解いた。終了3分前ACで全完はUnratedでもかなり嬉しい。
・内容としては雰囲気貪欲だから、あまり喜べるものじゃないけど。。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
A = LIST()
B = LIST()

ans1 = 0
for i in range(3):
    # あるAliceの手と、それで勝てるようなBobの手の小さい方
    ans1 += min(A[i], B[(i+1)%3])

ans2 = 0
for i in range(3):
    # あるAliceの手と、それで負けるやつとあいこになるやつを合わせた数の小さい方
    ans2 += min(A[i], B[i]+B[(i-1)%3])
# Aliceが勝てる最小数 = 全体 - Aliceが負けかあいこになる最大数
ans2 = N - ans2

print(ans2, ans1)
