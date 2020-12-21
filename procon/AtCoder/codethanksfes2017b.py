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

S=input()
N=len(S)

# 比較する部分文字列用の始点と終点2つずつ
l1=0
r1=l2=N//2
r2=N
if N%2==1:
    l2+=1

cnt=0
length=N
while True:
    # 前半の昇順と後半の逆順を比較する
    if S[l1:r1]==S[l2:r2][::-1]:
        break
    # 作られる予定の回文の長さの偶奇によって、indexを進める
    if length%2==0:
        l1+=1
        l2+=1
    else:
        l1+=1
        r1+=1
    length+=1
# 作られる最短の回文の長さ-元の長さ
print(length-N)
