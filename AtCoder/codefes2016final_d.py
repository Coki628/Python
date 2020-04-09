# -*- coding: utf-8 -*-

"""
・自力AC！実装頑張った。
・MODでまとめて、適切に貪欲。
・倍数ペアの方を先に作る。余ったやつで同値ペアを作る。
　ただし、偶数個の同値をなるべく多く残したいので、
　同値が奇数個だった場合は1個だけ優先して使うように移動させる。
"""

import sys
from collections import Counter

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

N, M = MAP()
A = LIST()

# MOD毎でまとめておく
modA = [a%M for a in A]
modC = Counter(modA)

# li[i] := mod Mでi毎に、(数値, 個数)のペアを個数の降順で格納したリスト
# ただし、奇数個ある数値は1つだけ優先して使うように後ろに移動させる
li = [Counter() for i in range(M)]
for a in A:
    li[a%M][a] += 1
for i in range(M):
    li[i] = li[i].most_common()
for i in range(M):
    tmp = []
    for j in range(len(li[i])):
        x, cnt = li[i][j]
        if cnt % 2 == 1:
            tmp.append((x, 1))
            li[i][j] = (x, cnt-1)
    li[i] += tmp

# mod Mでxの要素をcnt個消費する
def sub(x, cnt):
    while li[x] and cnt - li[x][-1][1] >= 0:
        cnt -= li[x][-1][1]
        li[x].pop()
    if li[x]:
        li[x][-1] = (li[x][-1][0], li[x][-1][1] - cnt)

# 0とM/2の場合は同じものでペアを作るので別処理
ans = 0
pcnt = modC[0] // 2
sub(0, pcnt*2)
ans += pcnt
if M % 2 == 0:
    pcnt = modC[M//2] // 2
    sub(M//2, pcnt*2)
    ans += pcnt

# それ以外は足してMになる同士でペアを作る
for i in range(1, ceil(M, 2)):
    j = M - i
    pcnt = min(modC[i], modC[j])
    sub(i, pcnt)
    sub(j, pcnt)
    ans += pcnt

# 倍数ペアを全て作ったら、残ったものから同値ペアを作る
for i in range(M):
    for _, cnt in li[i]:
        ans += cnt // 2
print(ans)
