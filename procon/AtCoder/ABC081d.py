# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc086/editorial.pdf
・累積和
・解法見たらそうだね！って感じだけど全然出なかった。
・というか、制約N=50に完全に頭が引っ張られて累積和のO(N)が全く予想出来なかった。
・添え字3つくらい使うメモ化再帰とかDPとか想像してしまった。
・この解法でいいならN=10万とかだよなー普通ー。
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

ans=[]
mn=min(A)
mnidx=A.index(mn)
mx=max(A)
mxidx=A.index(mx)
if mn<0 and mx>0:
    if abs(mx)>=abs(mn):
        for i in range(N):
            A[i]+=mx
            ans.append((mxidx+1, i+1))
        mn=min(A)
    else:
        for i in range(N):
            A[i]+=mx
            ans.append((mnidx+1, i+1))
        mx=max(A)

if mn>=0:
    for i in range(N-1):
        A[i+1]+=A[i]
        ans.append((i+1, i+2))
else:
    for i in range(N-1, 0, -1):
        A[i-1]+=A[i]
        ans.append((i+1, i))

print(len(ans))
for x, y in ans:
    print(x, y)
