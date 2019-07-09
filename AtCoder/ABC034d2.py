# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc034
・二分探索
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
WP=[]
for i in range(N):
    w,p=MAP()
    WP.append((w, p))

def calc(m):
    li=[]
    # 目指すパーセンテージが決まっていれば、量の違う各食塩水を同じ尺度で比較できる
    for i, wp in enumerate(WP):
        w,p=wp
        li.append(((p-m)*w, i))
    li.sort(reverse=True)
    ans=[0, 0]
    # 良かった方からK個使う
    for i in range(K):
        idx=li[i][1]
        w,p=WP[idx]
        # 全体
        ans[0]+=w
        # 塩
        ans[1]+=w*p/100
    # 今回の最善を尽くして、mパーセント以上を達成できるか
    return ans[1]/ans[0]*100>=m

ok=0
ng=100
# 100回くらいやれば誤差がほぼなくなる
for i in range(100):
    mid=(ok+ng)/2
    if calc(mid):
        ok=mid
    else:
        ng=mid
print(ok)
