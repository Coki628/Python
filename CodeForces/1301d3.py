# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/kiri8128/status/1227997444798988289
・行ける経路足りてなかったー。結局うまく回ると全部網羅できるのね。。
・愚直実装おつ、なやつ。
・さすがに冗長過ぎたので共通処理まとめて整理。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

H, W, K = MAP()

L = 'L'
R = 'R'
D = 'D'
U = 'U'
mx = 4*H*W - 2*H - 2*W
if mx < K:
    NO()
    exit()

def opr(dist, dir):
    global cnt
    if cnt - dist <= 0:
        ans.append((cnt, dir))
        cnt = 0
        return True
    else:
        ans.append((dist, dir))
        cnt -= dist
        return False

ans = []
cnt = K
i = 0
while i < W-1:
    if H > 1:
        if opr(H-1, D):
            break
        if opr(H-1, U):
            break
    if opr(1, R):
        break
    i += 1

i = 0
while cnt > 0 and i < H-1:
    if opr(1, D):
        break
    if W > 1:
        if opr(W-1, L):
            break
        if opr(W-1, R):
            break
    i += 1

if cnt > 0:
    for _ in range(1):
        if H > 1:
            if opr(H-1, U):
                break
        if W > 1:
            if opr(W-1, L):
                break

YES()
print(len(ans))
[print(a, b) for a, b in ans]
