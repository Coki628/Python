# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・ランレングス圧縮、縦横独立で考える、集合管理するDP
"""

import sys

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

S = input()
N = len(S)
X, Y = MAP()

# ランレングス圧縮
A1, A2 = [], []
cnt = d = 0
for s in S:
    if s == 'T':
        if d == 0:
            A1.append(cnt)
        else:
            A2.append(cnt)
        cnt = 0
        d ^= 1
    else:
        cnt += 1
if d == 0:
    A1.append(cnt)
else:
    A2.append(cnt)
# 最初だけ右固定なので別処理
X -= A1[0]
A1.pop(0)

# 縦横独立に、到達可の場所を確認する
M1 = len(A1)
M2 = len(A2)
dpx = [set() for i in range(M1+1)]
dpy = [set() for i in range(M2+1)]
dpx[0].add(0)
dpy[0].add(0)
for i, a in enumerate(A1):
    for x in dpx[i]:
        dpx[i+1].add(x+a)
        dpx[i+1].add(x-a)
for i, a in enumerate(A2):
    for y in dpy[i]:
        dpy[i+1].add(y+a)
        dpy[i+1].add(y-a)

if X in dpx[M1] and Y in dpy[M2]:
    Yes()
else:
    No()
