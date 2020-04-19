# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

K = INT()
S = set()
idxs = defaultdict(list)
Bs = []
for i in range(K):
    N = INT()
    A = LIST()
    sm = sum(A)
    # B[i] := Aの総和からA[i]を引いた数
    B = [0] * N
    for j, a in enumerate(A):
        B[j] = sm - A[j]
    Bs.append(B)
    for b in set(B):
        # 値bを使えるのはi番目の数列
        idxs[b].append(i)
        S.add(b)

# 値bについて全部見る
for b in S:
    # 2つ以上の数列で使われたものがあれば答えに使える
    if len(idxs[b]) >= 2:
        idx1 = idxs[b][0]
        idx2 = idxs[b][1]
        ans = []
        for i, b2 in enumerate(Bs[idx1]):
            if b2 == b:
                ans.append((idx1+1, i+1))
                break
        for i, b2 in enumerate(Bs[idx2]):
            if b2 == b:
                ans.append((idx2+1, i+1))
                break
        break
else:
    NO()
    exit()
YES()
print(*ans[0])
print(*ans[1])
