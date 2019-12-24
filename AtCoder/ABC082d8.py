# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・ランレングス圧縮、縦横独立で考える、集合管理するDP
・DPをbitset高速化。pythonTLE,pypyAC0.5秒だったのが、pythonAC0.06秒。
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
# 集合管理をbitset高速化
dpx = [0] * (M1+1)
dpy = [0] * (M2+1)
# 負数を考慮してずらしてから始める(制約8000だけど、最初を右固定補正してるから-16000まである)
offset = 20000
dpx[0] = 1 << offset
dpy[0] = 1 << offset
for i, a in enumerate(A1):
    dpx[i+1] |= dpx[i] << a
    dpx[i+1] |= dpx[i] >> a
for i, a in enumerate(A2):
    dpy[i+1] |= dpy[i] << a
    dpy[i+1] |= dpy[i] >> a

if dpx[M1] & 1<<(X + offset) and dpy[M2] & 1<<(Y + offset):
    Yes()
else:
    No()
