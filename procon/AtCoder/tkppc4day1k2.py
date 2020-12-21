# -*- coding: utf-8 -*-

"""
参考：https://drive.google.com/file/d/1CApCqwK7Hfgu81nqysUaPxSU2INI7l6R/view
・公式解
・優先度付きキュー
・なんか前も想定解が優先度付きキューの問題でindexセグ木使ったことあったな。。
・↑あった。ABC137Dだ。後ろからやるし、考え方は似てる。
・優先度付きキューが肝になる問題、気付けない率高いんだよなー。
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

N = INT()
A = [0] + LIST()

# まず経過日数より多い分は無意味なので削る
for i in range(N+1):
    A[i] = min(A[i], i)

que = []
heappush(que, -A[-1])
cur = N
ans = 0
while cur > 0:
    # 現在日より後ろで一番多く減らせる日を取得
    mx = -heappop(que)
    for _ in range(mx):
        cur -= 1
        # 現在日が戻ることで新たに使えるようになる日を追加していく
        heappush(que, -A[cur])
    ans += 1
print(ans)
