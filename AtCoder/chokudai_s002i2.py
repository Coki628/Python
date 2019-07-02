# -*- coding: utf-8 -*-

"""
・下位置換なやつを枝刈りしようとした→失敗
"""

import sys
from operator import itemgetter

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
food=[]
for i in range(N):
    a,b=MAP()
    food.append((a, b, i))

food2=[]
food.sort(key=itemgetter(0), reverse=True)
at=food[0][1]
food2.append(food[0])
for f in food:
    if f[1]>at:
        food2.append(f)
        at=f[1]

food3=[]
food2.sort(key=itemgetter(1), reverse=True)
hp=food2[0][0]
food3.append(food2[0])
for f in food2:
    if f[0]>hp:
        food3.append(f)
        hp=f[0]

def calc(A, B):
    a=ceil(A[0], B[1])
    b=ceil(B[0], A[1])
    if a>b:
        return A
    elif b>a:
        return B
    else:
        return ()

M=len(food3)
if M==1:
    print(food3[0][2]+1)
    exit()
score=[0]*N
for i in range(M):
    for j in range(i+1, M):
        res=calc(food3[i], food3[j])
        if len(res):
            score[res[2]]+=1
if max(score)==M-1:
    print(score.index(M-1)+1)
else:
    print(-1)
