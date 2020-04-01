# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/69464
・制約強い方。
・各値毎に必要な操作回数のリストをまとめておく。
・計算量がざっとNlogN個くらいの要素にソートかけてもう1つlogが乗るから、
　ちょっと重いかなーと思ったけど、pypyAC0.9秒。
"""

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

N, K = MAP()
A = LIST()

# vals[x] := 各aから値xを得るための操作回数を集計したリスト
vals = defaultdict(list)
for a in A:
    cnt = 0
    while a > 0:
        vals[a].append(cnt)
        a //= 2
        cnt += 1
    vals[0].append(cnt)

ans = INF
for li in vals.values():
    # 操作回数の少ない順でソート
    li.sort()
    if len(li) >= K:
        ans = min(ans, sum(li[:K]))
print(ans)
