# -*- coding: utf-8 -*-

"""
・400点自力AC
・計算量心配だったけどpypyで0.9秒
・外ループ*約数列挙=O(N*√N)=20万*450くらい=9000万なんだけど、
　約数列挙やるのはボール入れる必要がある時だけだし、外ループが進むにつれ
　やる範囲も小さくなるから、実際はもっとずっと少ないはず。
"""

import sys
from math import sqrt

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

def divisor_set(N: int) -> set:
    """ 約数の列挙・個数 """
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N=INT()
A=LIST()

# その位置が偶奇どちらになるかを保持するA2
A2=[0]*N
B=[]
# 後ろから決めていく
for i in range(N-1, -1, -1):
    # ボールを入れるべきかどうか
    if A[i]!=A2[i]:
        B.append(i+1)
        # 約数列挙
        div=divisor_set(i+1)
        for num in div:
            # 自分の変化が影響する所を全部変える
            A2[num-1]=1-A2[num-1]
        
print(len(B))
print(*B)
