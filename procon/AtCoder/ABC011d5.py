# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc011
　　　http://baitop.hatenadiary.jp/entry/2018/09/03/164030
・組み合わせ系、確率系
・上下の個数と左右の個数を固定、これでXとYを独立に考えられる。
・上下左右全体の組み合わせと、上下と左右のうまくいく組み合わせの3つを最後にかけ合わせる。
・想定解は値が大きくなり過ぎるのを防ぐために、組み合わせ数を都度確率にして足してたぽいけど、
　Pythonはその辺の精度とか優秀なのか、最後に一括で確率に、でも問題なく通ってしまった。
・一応、確率にして足す版もやってみた。速度、変わらず。
"""

import sys
from math import factorial

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

def nCr(n, r):
    """ 事前テーブルなし組み合わせ簡易版、確率を返す版(2**nで割るだけ) """
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return (factorial(n) // (factorial(r) * factorial(n-r))) / 2**n

N,D=MAP()
X,Y=MAP()

# Dを1として簡略化(割り切れない場所には辿り着けない)
if X%D!=0 or Y%D!=0:
    print(0)
    exit()
X//=D
Y//=D

scs=0
for i in range(N+1):
    j=N-i
    # 左右にi回、上下にj回動く全組み合わせの数(nCr(N, j)でもいい)
    total=nCr(N, i)

    # i回左右に動いてもXには辿り着けない
    if i%2!=X%2 or i<abs(X): continue
    # ちょうどXに辿り着けるような左の回数
    r=(i-X)//2
    # i回のうち、左をr回選ぶ組み合わせの数
    scsx=nCr(i, r)
    
    # j回上下に動いてもYには辿り着けない
    if j%2!=Y%2 or j<abs(Y): continue
    # ちょうどYに辿り着けるような下の回数
    r=(j-Y)//2
    # j回のうち、下をr回選ぶ組み合わせの数
    scsy=nCr(j, r)

    # 3つの組み合わせの数をかけ合わせる
    scs+=total*scsx*scsy

print(scs)
