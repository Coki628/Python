# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/past202004-open/editorial.pdf
　　　https://rsk0315.hatenablog.com/entry/2020/05/02/181650
・みんな言ってた通り、これが1番きつかったね。。
・周期、ダブリング、二分探索
・設定複雑で実装大変。まあ天才考察より全然いいけど。
"""

import sys
from collections import defaultdict
from bisect import bisect_left

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

D, L, N = MAP()
C = [c-1 for c in LIST()]
KFT = []
for i in range(N):
    k, f, t = MAP()
    k -= 1
    f -= 1
    KFT.append((k, f, t))

# food[i] := 料理iの出現indexのリスト
food = defaultdict(list)
MAX = 32
# nxt[0][i] := 位置iの料理が次に提供されるindex
nxt = list2d(MAX, D, -1)
for i in range(D*2):
    if food[C[i%D]]:
        previ = food[C[i%D]][-1]
        nxt[0][previ%D] = i % D
    food[C[i%D]].append(i)

# cnt[0][i] := 位置iの料理を食べた後、次に同じ料理を食べるまでに食べる総数
cnt = list2d(MAX, D, 0)
for i in range(D):
    dist = (nxt[0][i] - i) % D if (nxt[0][i] - i) % D != 0 else D
    cnt[0][i] = ceil(dist, L)

# ダブリングのテーブル構築
for k in range(1, MAX):
    for i in range(D):
        if cnt[k-1][i] < INF:
            nxt[k][i] = nxt[k-1][nxt[k-1][i]]
            cnt[k][i] = cnt[k-1][i] + cnt[k-1][nxt[k-1][i]]

for m, f, t in KFT:
    if not food[m]:
        print(0)
        continue
    # 初期位置から周期に入るまでの調整
    idx = food[m][bisect_left(food[m], f)]
    total = ceil(idx-f, L) + 1
    idx %= D
    if total > t:
        print(0)
        continue
    fav = 1
    # 周期に入ったらダブリングで数える
    for k in range(MAX-1, -1, -1):
        if total + cnt[k][idx] <= t:
            total += cnt[k][idx]
            fav += 2**k
            idx = nxt[k][idx]
    print(fav)
