# -*- coding: utf-8 -*-

"""
・1つ目のコードから考察を進めて整理。
・現在位置を1番目から順に進めれば、常にC2[cur]で未出現を1つにできて、それが新しく追加する値。
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

C = Counter()
C2 = defaultdict(list)
for i in range(N-2):
    q = LIST()
    q.sort()
    a, b, c = q
    C[a] += 1
    C[b] += 1
    C[c] += 1
    C2[a].append((b, c))
    C2[b].append((a, c))
    C2[c].append((a, b))

cur = 0
ans = []
used = set()
# まず単体の出現回数から、最初の2つを決める
for k, v in C.items():
    # 出現回数1回が先頭
    if v == 1:
        cur = k
        ans.append(k)
        used.add(k)
        # どちらかは出現回数2回のはずで、そっちが2番目
        for a, b in C2[cur]:
            if C[a] == 2:
                ans.append(a)
                used.add(a)
            else:
                ans.append(b)
                used.add(b)
        break

# 残りは集計しておいた値から、出現済かどうかと照らし合わせながら特定していける
while len(ans) < N:
    # 現在位置を1番目から順に進めれば、常にC2[cur]で未出現を1つにできて、それが新しく追加する値
    for a, b in C2[cur]:
        if a not in used:
            used.add(a)
            ans.append(a)
            cur = b
            break
        elif b not in used:
            used.add(b)
            ans.append(b)
            cur = a
            break
print(*ans)
