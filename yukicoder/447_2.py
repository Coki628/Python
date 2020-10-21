"""
・Decimalのとこ、Fractionにしてみた。ちょっと速くなって0.32秒。
・あと、みんな誤差どうしてるか見たら、10倍してたり(それはそう)、
　切り捨て処理を途中じゃなくて最後にしたりして、ACしてた。
"""

import sys
from collections import defaultdict
from operator import itemgetter
from fractions import Fraction

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
    return L[p]*50 + L[p]*50 // (Fraction(4, 5)+Fraction(1, 5)*prank[p])

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
