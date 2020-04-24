# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/abc127/submissions/me
・放置してたやつ、久しぶりに着手したけど自力ならず。
・2つのリストで半分ずつ持って真ん中を効率良く管理、っていう方針は
　AC062dと通じるものを感じた。
・C++でmultisetでやっていたものをpythonに移植。
　参照する要素が最大or最小だけでいいのなら、heapqで代用できる。
・pythonAC0.9秒、pypyAC0.9秒。pypyはheapqあんま速くないんかな。
"""

import sys
from heapq import heappush, heappop

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

Q = INT()
_, a, b = MAP()
mid = a
asm = 0
bsm = b
ln = 1
que1, que2 = [-a], [INF]
for _ in range(Q-1):
    li = LIST()
    if li[0] == 1:
        a, b = li[1:]
        bsm += b
        l = -que1[0]
        r = que2[0]
        if ln % 2 == 1:
            asm += abs(a - l)
            if a < mid:
                heappush(que1, -a)
                # 今回の要素を入れてから、その中での最大値を大きい方へ移す
                heappush(que2, -que1[0])
                heappop(que1)
            else:
                heappush(que2, a)
        else:
            if a < l:
                asm += abs(a - l)
            elif a > r:
                asm += abs(a - r)
            if a > mid:
                heappush(que2, a)
                # 今回の要素を入れてから、その中での最小値を小さい方へ移す
                heappush(que1, -que2[0])
                heappop(que2)
            else:
                heappush(que1, -a)
        ln += 1
        mid = -que1[0]
    else:
        print(mid, asm + bsm)
