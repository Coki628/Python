# -*- coding: utf-8 -*-

"""
参考：https://drive.google.com/drive/folders/0BziHwCcP5FcyejdCaFVibmlidFE
・公式解を参考に前からfor文でやってみる
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

N,L=MAP()
# 小数を考慮するため、2倍して考えておく
L*=2
A=[INT()*2 for i in range(N)]

if N==1:
    YES()
    exit()

# 跳べない区間が終わって跳べるようになる座標p
# 今回跳ぶ予定の(複数を含む)ハードルの区間の合計l
l=p=0
for i in range(N-1):
    # 前のハードルと次のハードルとの区間
    gap=A[i+1]-A[i]
    # 区間=Lは越えれないし跳べない区間も確保できないからNG
    if gap==L:
        NO()
        exit()
    # 区間が短い間は空中にいる(この区間で着地してしまうと次跳べない)
    elif gap<L:
        l+=gap
        if l>=L:
            NO()
            exit()
    # 区間が長い所で着地してもう1回跳ぶ
    elif gap>L:
        # 初回だけ跳べない区間がないことを考慮
        if p==0:
            # A[i]とA[i+1]の間の着地点
            land=max(p+L, A[i]+1)
        else:
            land=max(p-L, A[i]+1)
        # A[i+1]に向かって跳ぶ地点
        jmp=max(land+L, A[i+1]+1-L)
        # 跳ぶ地点がA[i+1]以上だとぶつかるからNG
        if jmp>=A[i+1]:
            NO()
            exit() 
        # 次以降のためにpとlを更新
        p=jmp+L*2
        l=0
# 最後までいけばOK
YES()
