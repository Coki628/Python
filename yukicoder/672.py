"""
・自力AC
・累積和、zero sum ranges系
・最初方針ミスって危なかった。
・AとBが同じ数になればいいから、+1,-1で累積和して差が0になる区間から最大を取る。
・すげーよく見る典型なんだけど、zero sum ranges系よく見落とすから、ちゃんと気付けてよかった。
　欲を言えば、もっと秒で気付けるようにならないと。。
"""

import sys
from itertools import accumulate

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

S = input()
N = len(S)

acc = [0] * N
for i in range(N):
    if S[i] == 'A':
        acc[i] = 1
    else:
        acc[i] = -1
acc = [0] + list(accumulate(acc))

D = {}
ans = 0
for i, a in enumerate(acc):
    if a in D:
        ans = max(ans, i - D[a])
    else:
        D[a] = i
print(ans)
