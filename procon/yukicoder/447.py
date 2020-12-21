"""
・自力AC
・実装系
・こういうの、解けるんだからもっと手早くできるようになりたいね。
・整ったと思ったらWAでしばらく詰まってたけど、結局小数誤差だった。。
・Decimalで通ったけど、制約T=4000で0.55秒はやっぱ遅いなぁ。
"""

import sys
from collections import defaultdict
from operator import itemgetter
from decimal import Decimal

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
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
L = LIST()
T = INT()
last = defaultdict(lambda: -1)
scores = defaultdict(lambda: [0]*N)
prank = [1] * N

def calc(p):
    return L[p]*50 + L[p]*50 // (Decimal('0.8')+Decimal('0.2')*prank[p])

for i in range(T):
    name, p = input().split()
    p = ord(p) - 65
    scores[name][p] = calc(p)
    last[name] = i
    prank[p] += 1

M = len(scores)
info = []
for name, li in scores.items():
    info.append((sum(li), last[name], name, li))
info.sort(key=itemgetter(1))
info.sort(key=itemgetter(0), reverse=1)

for i, a in enumerate(info):
    print(i+1, a[2], *a[3], a[0])
