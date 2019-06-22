# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day2/editorial-F.pdf
　　　http://fluffyowl.hatenablog.com/entry/2019/05/04/045832
・公式解
・ミニマックス法でメモ化再帰
・パターンでいけると思ったら意外と詰まって自力実装できなかった…。
・a1とa2は同じ箱に対しての結果だから、個別にやらないで戻り値同士足す。
・+100円とかは戻り値に対して、確率かける前に足し込む。
・後手番の時は金額足さないけど、確率はちゃんとかける。
・等々。。
・pypyでもTLE。多分、python的に6次元配列がつらいのかと。
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
b1,b2=MAP()
c1,c2=MAP()

memo=[[[[[[-1] * 11 for _ in range(11)] for _ in range(11)] for _ in range(11)] for _ in range(11)] for _ in range(11)]
turn=(a1+a2+b1+b2+c1+c2)%2
def rec(a1, a2, b1, b2, c1, c2):
    if memo[a1][a2][b1][b2][c1][c2]!=-1:
        return memo[a1][a2][b1][b2][c1][c2]
    if a1<0 or a2<0 or b1<0 or b2<0 or c1<0 or c2<0:
        return 0
    if a1+a2+b1+b2+c1+c2==0:
        return 0
    global turn
    # 先手番
    if (a1+a2+b1+b2+c1+c2)%2==turn:
        ans=-INF
        if a1+a2>0:
            res=(rec(a1-1, a2, b1, b2, c1, c2)+100)*a1/(a1+a2)+(rec(a1, a2-1, b1, b2, c1, c2)+50)*a2/(a1+a2)
            ans=max(ans, res)
        if b1+b2>0:
            res=(rec(a1, a2, b1-1, b2, c1, c2)+100)*b1/(b1+b2)+(rec(a1, a2, b1, b2-1, c1, c2)+50)*b2/(b1+b2)
            ans=max(ans, res)
        if c1+c2>0:
            res=(rec(a1, a2, b1, b2, c1-1, c2)+100)*c1/(c1+c2)+(rec(a1, a2, b1, b2, c1, c2-1)+50)*c2/(c1+c2)
            ans=max(ans, res)
    # 後手番
    else:
        ans=INF
        res=-1
        if a1+a2>0:
            res=rec(a1-1, a2, b1, b2, c1, c2)*a1/(a1+a2)+rec(a1, a2-1, b1, b2, c1, c2)*a2/(a1+a2)
        if res!=-1:
            ans=min(ans, res)
        if b1+b2>0:
            res=rec(a1, a2, b1-1, b2, c1, c2)*b1/(b1+b2)+rec(a1, a2, b1, b2-1, c1, c2)*b2/(b1+b2)
        if res!=-1:
            ans=min(ans, res)
        if c1+c2>0:
            res=rec(a1, a2, b1, b2, c1-1, c2)*c1/(c1+c2)+rec(a1, a2, b1, b2, c1, c2-1)*c2/(c1+c2)
        if res!=-1:
            ans=min(ans, res)
    memo[a1][a2][b1][b2][c1][c2]=ans
    return ans

print(rec(a1, a2, b1, b2, c1, c2))
