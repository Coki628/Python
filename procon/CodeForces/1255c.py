# -*- coding: utf-8 -*-

"""
・ほぼ自力ACだけど、時間かかったし参戦中に通せたかは分からない。
・自分でも呆れるくらい謎なdict祭りになってしまって、絶対もっと整理できると思った。
"""

import sys
from collections import defaultdict, Counter

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

C1 = defaultdict(list)
C2 = defaultdict(list)
C3 = Counter()
for i in range(N-2):
    q = LIST()
    q.sort()
    a, b, c = q
    C1[(a, b)].append(c)
    C1[(a, c)].append(b)
    C1[(b, c)].append(a)
    C2[a].append((b, c))
    C2[b].append((a, c))
    C2[c].append((a, b))
    C3[a] += 1
    C3[b] += 1
    C3[c] += 1

cur = 0
ans = []
used = set()
# まず単体の出現回数から、最初の2つを決める
for k, v in C3.items():
    if v == 1:
        cur = k
        ans.append(k)
        used.add(k)
        for a, b in C2[cur]:
            if C3[a] == 2:
                ans.append(a)
                used.add(a)
                cur = (min(k, a), max(k, a))
            else:
                ans.append(b)
                used.add(b)
                cur = (min(k, b), max(k, b))
        break

# 残りは集計しておいた値から、出現済かどうかと照らし合わせながら特定していける
i = 2
while True:
    for a in C1[cur]:
        if a not in used:
            cur = a
            used.add(a)
            ans += [a]
    i += 1
    if i >= N:
        break
    if len(C2[cur]) == 3:
        C4 = Counter()
        for a, b in C2[cur]:
            C4[a] += 1
            C4[b] += 1
        for a, b in C2[cur]:
            if a not in used and b not in used:
                cur = (a, b)
                used.add(a)
                used.add(b)
                if C4[b] == 2:
                    a, b = b, a
                ans += [a, b]
                break
        i += 2
    else:
        for a, b in C2[cur]:
            if a not in used:
                ans += [a]
                break
            if b not in used:
                ans += [b]
                break
        i += 1
    if i >= N:
        break
print(*ans)
