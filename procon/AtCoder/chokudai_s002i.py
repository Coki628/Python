# -*- coding: utf-8 -*-

"""
・貪欲に攻めようとした→失敗
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
food=[]
for i in range(N):
    a,b=MAP()
    food.append((a, b, i+1))

def calc(A, B):
    a=ceil(A[0], B[1])
    b=ceil(B[0], A[1])
    if a>b:
        return A
    elif b>a:
        return B
    else:
        return ()

cur=food[0]
draws=[]
i=1
while i<N:
    nxt=food[i]
    res=calc(cur, nxt)
    if len(res):
        cur=res
        i+=1
    else:
        draws.append(cur)
        draws.append(nxt)
        i+=2
        if i-1<N:
            cur=food[i-1]
        else:
            cur=()

if not len(cur):
    print(-1)
    exit()

for draw in draws:
    res=calc(cur, draw)
    if res[2]!=cur[2]:
        print(-1)
        exit()
print(cur[2])
