"""
・自力AC
・場合分け
・やるだけ、なら緑diffはないよなー、と思いつつ投げたらやっぱりWA。で、しばし考察。
・文字を他の色に1回退避させるつもりだったけど、穴の数によっては背景と穴を退避させて
　文字の方を1回で変えた方がいい場合もある。ということで、両方数えて最適な方にする。
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
EPS = 10 ** -10

D = {'0': 1, '1': 0, '2': 0, '3': 0, '4': 1, '5': 0, '6': 1, '7': 0, '8': 2, '9': 1}

S = input()
cnt1 = 1
for s in S:
    cnt1 += D[s]
cnt2 = len(S)
ans = min(cnt1, cnt2) * 2 + max(cnt1, cnt2)
print(ans)
