# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/diverta2019/editorial.pdf
・200点自力ならず。最低ー。
・式変形
　Rx+Gy+Bz=N => z=(N-Rx-Gy)/B
・zについて解けば、xとyのループ回して全検索できる系。
　全然知ってるアプローチだったのに出なかったっていうダメダメなやつ。
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

R,G,B,N=MAP()

cnt=0
for x in range(3001):
    for y in range(3001):
        z=(N-R*x-G*y)/B
        # zが0以上の整数になっていればOK
        if z>=0 and z.is_integer():
            cnt+=1
print(cnt)
