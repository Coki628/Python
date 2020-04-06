# -*- coding: utf-8 -*-

"""
・終了約5分前で自力AC！
・適切に貪欲
・最初、伸ばせるだけ伸ばした状態から縮める方針でやったらバグらせまくって死にかけて、
　頭切り替えて、一番縮めた状態から必要なとこまで伸ばす、にしたら、
　ほぼスクラッチから書き直したのに20分くらいで仕上がってAC。
・最初から正しい方針でいけてれば一番よかったけど、適切な判断で方針切り替えられたからまあよかったかな。
"""

import sys
from itertools import accumulate

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
A = LIST()

# そもそも足りないならNG
if sum(A) < N:
    print(-1)
    exit()

# まず一番縮めた状態を作る
B = [0] * M
for i in range(M):
    B[i] = i + A[i]
mx = max(B)
# この時点で長過ぎたらNG
if mx > N:
    print(-1)
    exit()

C = [0] * M
for i in range(1, M):
    # この位置を最大で伸ばせる長さ
    move = B[i-1] - 1
    # N未満な限り伸ばす
    if mx + move < N:
        mx += move
        C[i] = move
    else:
        # 最後はちょうどよくやる
        need = N - mx
        C[i] = need
        break
acc = list(accumulate(C))

ans = [0] * M
for i in range(M):
    ans[i] = i + 1 + acc[i]
print(*ans)
