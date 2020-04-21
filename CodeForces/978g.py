# -*- coding: utf-8 -*-

"""
・優先度付きキュー、スケジュール管理
・日程を1日ずつ進めて、使えるようになったやつからキューに追加。
　引き出しの優先度は締め切りの早いやつ。
　準備の間に合わなかった時があったらNG終了。
・N<=100の制約が、貪欲無理で2次元+遷移の3重ループのDPみたいなので頑張らなきゃ
　いけない感を醸し出してたけど、無事貪欲にAC。
・制約10万でもいけるやつだと思うけど、なんで100にしたんだろー。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
prep = [[] for i in range(N)]
need = [0] * M
exam = {}
for i in range(M):
    s, d, c = MAP()
    s -= 1; d -= 1
    need[i] = c
    prep[s].append((d, i))
    exam[d] = i

que = []
ans = [0] * N
for day in range(N):
    for d, i in prep[day]:
        heappush(que, (d, i))
    if day in exam:
        i = exam[day]
        if need[i] > 0:
            print(-1)
            exit()
        ans[day] = M + 1
    else:
        if que:
            d, i = heappop(que)
            need[i] -= 1
            if need[i] > 0:
                heappush(que, (d, i))
            ans[day] = i + 1
print(*ans)
