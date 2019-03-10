# -*- coding: utf-8 -*-

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

h1,m1=map(int, input().split(':'))
h2,m2=map(int, input().split(':'))

time1=h1*60+m1
time2=h2*60+m2
time3=(time1+time2)//2

h3=time3//60
m3=time3%60

print(str(h3).zfill(2)+':'+str(m3).zfill(2))
