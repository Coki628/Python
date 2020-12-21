# -*- coding: utf-8 -*-

"""
・500点自力AC
・期待値でこれは嬉しい。
・と思ったけど想定解見たら結構違くて、いわゆるゲームのメモ化再帰。
・何回何を引いても、結局単発の期待値は変わらないっていう考察は合ってるのかなー。
　嘘解法じゃないと信じたい。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

a1,a2=MAP()
asm=a1+a2
b1,b2=MAP()
bsm=b1+b2
c1,c2=MAP()
csm=c1+c2

# 各箱の1回引く分の期待値を出しておく
A=100*a1/asm if asm!=0 else 0
A+=50*a2/asm if asm!=0 else 0
B=100*b1/bsm if bsm!=0 else 0
B+=50*b2/bsm if bsm!=0 else 0
C=100*c1/csm if csm!=0 else 0
C+=50*c2/csm if csm!=0 else 0

box=[[A, asm], [B, bsm], [C, csm]]
box.sort(reverse=True)

# 期待値のおいしい箱から貪欲に引いていく
butsu=hira=turn=0
for i in range(3):
    for j in range(box[i][1]):
        if turn%2==0:
            butsu+=box[i][0]
        else:
            hira+=box[i][0]
        turn+=1
print(butsu)
