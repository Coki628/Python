# -*- coding: utf-8 -*-

"""
・色々頑張って自力実装したけどTLE
・多分、毎回全部からmin取ってるのがしんどい。
・最小値取得の回数減らしたりちょっと最適化したけど惜しくも1ケースTLEなので、
　C++でほぼ同じコード書いて提出。案外危なげな1.5秒AC。
・大変だったけど500点自力ACは嬉しい。
"""

import sys
from collections import Counter, defaultdict

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
INF = float('inf')
MOD = 10 ** 9 + 7

N, Q = MAP()
X = set()
A = defaultdict(list)
for i in range(N):
    s, t, x = MAP()
    # 全ての工事を同じ時系列に持ってくる
    X.add(s-x)
    X.add(t-x)
    A[s-x].append((x, 0))
    A[t-x].append((x, 1))
D = LIST(Q)

Dset = set(D)
S = sorted(X|Dset)
B = {}
C = Counter()
# 人か工事のある箇所だけ見る
for key in S:
    # 工事の状態に変化のある時
    if key in A:
        for x, flag in A[key]:
            # 工事の本来の開始時期毎に値を増減させる
            if flag == 0:
                C[x] += 1
            else:
                C[x] -= 1
            if C[x] == 0:
                del C[x]
    # 人が出発する時
    if key in Dset:
        # この時点で存在する工事の最も早いものを使いたいので、最小値を取得
        # 工事が1つもなければその人は止まらない
        mn = min(C) if len(C) else -1
        B[key] = mn
[print(B[x]) for x in D]
