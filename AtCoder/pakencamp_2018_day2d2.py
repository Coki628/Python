# -*- coding: utf-8 -*-

"""
・自力WAから色々試行錯誤の末AC。
・ランレングス圧縮、二分探索、大まかな方針は合ってた。
・でもそんな本質じゃないところで色々バグってた。。以下、その概要。
・左追加で色が変わらない時にl_offsetずらしてなかった。
・時系列順にクエリが来てるとは限らないのにクエリ順じゃなくて時系列順のまま出力してた。
・同じ時間へのクエリが1つとは限らないのにdictに値詰めてた。
"""

import sys
from collections import deque, defaultdict
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
S = input()
M = INT()
DF = []
for _ in range(M):
    d, f = input().split()
    DF.append((d, f))
# 同じ時間への複数クエリもありえるのでvalueはリストにする
TP = defaultdict(list)
for i in range(INT()):
    t, p = MAP()
    TP[t-1].append((i, p - 1))

# ランレングス圧縮(位置取得したいので持たせるのは長さじゃなくindex)
idxs, colors = deque(), deque()
cur = S[0]
idx = 0
for i in range(1, N):
    if cur != S[i]:
        idxs.append(idx)
        colors.append(cur)
        cur = S[i]
        idx = i
idxs.append(idx)
colors.append(cur)

ans = []
# indexの変化を管理する
l_offset = 0
r_offset = N-1
for i in range(M):
    d, f = DF[i]
    # 左に追加
    if d == 'L':
        l_offset += 1
        if f != colors[0]:
            # 種類が1つだけの時は反対の色でも挟めないので追加
            if len(idxs) == 1:
                colors.appendleft(f)
                idxs.appendleft(-l_offset)
            # 他は挟めるので、色の区間が1つ減る
            else:
                colors.popleft()
                idxs.popleft()
        # 左端のindexは左追加時は常に動かす
        idxs[0] = -l_offset
    # 右に追加
    else:
        r_offset += 1
        if f != colors[-1]:
            if len(idxs) == 1:
                colors.append(f)
                idxs.append(r_offset)
            else:
                colors.pop()
                idxs.pop()
    # クエリがある時
    if i in TP:
        for j, p in TP[i]:
            target = p - l_offset
            # offsetでindexを合わせて、二分探索で位置取得
            idx = bisect_right(idxs, target) - 1
            ans.append((j, colors[idx]))

# 時系列順になっているので、クエリの順にソートし直す
ans.sort()
[print(color) for i, color in ans]
