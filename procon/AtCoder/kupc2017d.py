# -*- coding: utf-8 -*-

"""
参考：http://www.kupc.jp/static/media/D.6fb71823.pdf
　　　http://kazune-lab.net/contest/2017/10/01/kupc2017/
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

def calc(n):
    cnt=0
    # 四方にふた回り大きいグリッドを作る
    grid=list2d(n+4, n+4, '*')
    # ビット全探索で各マス1or2を全通り試す
    for i in range(1<<(n*n)):
        for j in range(n*n):
            x=j%n
            y=j//n
            if i>>j&1:
                grid[y+2][x+2]='2'
            else:
                grid[y+2][x+2]='1'
        if check(grid, n):
            cnt+=1
    return cnt

def check(grid, n):
    for j in range(2,n+2):
        for k in range(2, n+2):
            # 横
            a=grid[j][k]
            b=grid[j][k+1]
            c=grid[j][k+2]
            if a==b==c=='1' or a==b==c=='2':
                return False
            # 縦
            a=grid[j][k]
            b=grid[j+1][k]
            c=grid[j+2][k]
            if a==b==c=='1' or a==b==c=='2':
                return False
            # 斜め(左上→右下)
            a=grid[j][k]
            b=grid[j+1][k+1]
            c=grid[j+2][k+2]
            if a==b==c=='1' or a==b==c=='2':
                return False
            # 斜め(右上→左下)
            a=grid[j][k]
            b=grid[j+1][k-1]
            c=grid[j+2][k-2]
            if a==b==c=='1' or a==b==c=='2':
                return False
    return True

N=INT()

if N<=2:
    print(1, 1)
elif 3<=N<=4:
    # N=3,4だけ全探索
    print(2, calc(N))
else:
    # 以降は4*縦横=8通りで循環するだけ
    print(2, 8)
