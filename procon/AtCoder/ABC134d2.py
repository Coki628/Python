# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc134/editorial.pdf
　　　https://www.youtube.com/watch?v=-o4r74eJzV4
・公式解
・約数でなく、倍数で探しに行けばN√NじゃなくてNlogNでいける。
・やっぱこっちのが速くて、pypyで0.3秒。
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

N=INT()
A=LIST()

# その位置にボールを入れるかを保持するA2
A2=[0]*N
B=[]
# 後ろから決めていく
for i in range(N-1, -1, -1):
    cur=i+1
    cnt=0
    # curの倍数を全部見に行く
    for j in range(cur*2, N+1, cur):
        cnt^=A2[j-1]
    # ボールを入れるべきかどうか
    if A[i]!=cnt:
        B.append(cur)
        A2[i]=1

print(len(B))
print(*B)
