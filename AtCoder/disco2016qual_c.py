# -*- coding: utf-8 -*-

import sys
from math import sqrt
from collections import Counter

def input(): return sys.stdin.readline().strip()
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

# 素因数分解
def factorize(num: int) -> dict:
    d = Counter()
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

N,K=MAP()
A=LIST()

facts=[None]*N
kfact=factorize(K)
for i in range(N):
    facts[i]=factorize(A[i])

B=[]
ans=0
al=N-1
for i in range(N):
    ok=no=0
    for k, v in kfact.items():
        if facts[i][k]>=v:
            ok+=1
        elif facts[i][k]==0:
            no+=1
    if ok==len(kfact):
        ans+=al
        al-=1
    elif no==len(kfact):
        pass
    else:
        B.append(i)

for i in range(len(B)):
    for j in range(i+1, len(B)):
        c=Counter()
        c.update(facts[B[i]])
        c.update(facts[B[j]])
        ok=0
        for k, v in kfact.items():
            if c[k]>=v:
                ok+=1
        if ok==len(kfact):
            ans+=1
print(ans)
