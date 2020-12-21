# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc128/editorial.pdf
・400自力ならず。
・メモ化再帰とか考えてしまって、全然違った。
・解法とか参考にして作れたけど、右から取るループのカウントが取りづらかった。
・分かってしまえば、特別なことはしてないんだよね。。
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

N,K=MAP()
V=LIST()

def calc(l, r):
    # Kより多く操作するケースは不可
    if l+(N-r)>K:
        return -INF
    ngtv=[]
    res=0
    # 左から取る分
    for i in range(l):
        res+=V[i]
        if V[i]<0:
            ngtv.append(V[i])
    # 右から取る分
    for i in range(N-1, r-1, -1):
        res+=V[i]
        if V[i]<0:
            ngtv.append(V[i])
    # マイナス分をソートしておく
    ngtv.sort()
    # 取った分の操作回数
    k=l+(N-r)
    i=0
    # まだ操作できるかつマイナスが残ってる限り、捨てる
    while k<K and i<len(ngtv):
        res-=ngtv[i]
        i+=1
        k+=1
    return res

ans=-INF
# 左右どこまで取るかを全通り試す
for l in range(N+1):
    for r in range(l, N+1):
        ans=max(ans, calc(l, r))
print(ans)
