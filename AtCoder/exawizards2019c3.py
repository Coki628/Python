# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/exawizards2019/editorial.pdf
・二分探索
・20万でO(QlogN)だけど、python1.9秒ギリギリ、pypyは0.5秒くらいで余裕だった。
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

N,Q=MAP()
S=input()
TD=[input().split() for i in range(Q)]

# ある位置にいるゴーレムを動かして、左端から外に出ないか判定する
def calc1(m):
    idx=m
    s=S[idx]
    for i in range(Q):
        t,d=TD[i]
        if t==s:
            if d=='L':
                idx-=1
            elif d=='R':
                idx+=1
        if idx<0:
            return False
        if idx>=N:
            return True
        s=S[idx]
    return True

# ある位置にいるゴーレムを動かして、右端から外に出ないか判定する
def calc2(m):
    idx=m
    s=S[idx]
    for i in range(Q):
        t,d=TD[i]
        if t==s:
            if d=='L':
                idx-=1
            elif d=='R':
                idx+=1
        if idx>=N:
            return False
        if idx<0:
            return True
        s=S[idx]
    return True

low=-1
hi=N
while low+1<hi:
    mid=(hi+low)//2
    if calc1(mid):
        hi=mid
    else:
        low=mid
# 左端にたどり着かない(生き残る)ゴーレムのうち、初期位置が最も左のもの
lowest=hi

low=-1
hi=N
while low+1<hi:
    mid=(hi+low)//2
    if calc2(mid):
        low=mid
    else:
        hi=mid
# 右端にたどり着かない(生き残る)ゴーレムのうち、初期位置が最も右のもの
highest=low

print(max(highest-lowest+1, 0))
