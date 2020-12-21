"""
・自力AC！
・累積和、二分探索
・こういう典型は数学っぽくないから解きやすいよね。
・計算量2000*2000=400万にlog乗ってるけど、pypyAC0.6秒。
"""

import sys
from itertools import accumulate
from bisect import bisect_right

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

N = INT()
S = input()
A = LIST()
M = INT()
Q = LIST()

enemy = [0] * N
for i, s in enumerate(S):
    if s == 'E':
        enemy[i] = 1
enemy = [0] + list(accumulate(enemy))
acc = [0] + list(accumulate(A))

for k in Q:
    ans = 0
    for l in range(N):
        r = bisect_right(acc, acc[l]+k, lo=l) - 1
        cnt = enemy[r] - enemy[l]
        ans = max(ans, cnt)
    print(ans)
