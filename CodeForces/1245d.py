# -*- coding: utf-8 -*-

"""
・自力WA
"""

import sys
from operator import itemgetter

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

N = INT()
cities = []
for i in range(N):
    x, y = MAP()
    cities.append((x, y))
C = LIST()
K = LIST()
for i in range(N):
    c = C[i]
    k = K[i]
    x, y = cities[i]
    cities[i] = (i, x, y, c, k)

cities.sort(key=itemgetter(3))
total = 0
stations = []
edges = []
done = [0] * N
for i in range(N):
    id1, x1, y1, c1, k1 = cities[i]
    if not done[id1]:
        stations.append(id1+1)
        total += c1
        done[id1] = 1

        for j in range(N):
            id2, x2, y2, c2, k2 = cities[j]
            if not done[id2]:
                dist = abs(x1-x2) + abs(y1-y2)
                if dist * (k1+k2) < c2:
                    edges.append((id1+1, id2+1))
                    total += dist * (k1+k2)
                    done[id2] = 1

print(total)
print(len(stations))
print(*stations)
print(len(edges))
for a, b in edges:
    print(a, b)
