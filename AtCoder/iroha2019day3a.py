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

a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z=MAP()

print(a-b)
print(c+d)
print(max(0, f-e+1))
print((g+h+i)//3+1)

# 辞書順最小の部分文字列
def findmin(S, l):
    n=len(S)
    cnt=idx=0
    used=[False]*n
    # そこより先で最小の文字を探す、を末尾までやる
    while idx<n:
        mn='{'
        for j, s in enumerate(S[idx:]):
            j+=idx
            if s<mn:
                mn=s
                idx=j
            if idx==n:
                break
        used[idx]=True
        cnt+=1
        idx+=1
        if cnt==l:
            break
    # 上記処理で使わなかった文字を後ろから順に足す
    if cnt<l:
        for i in range(n-1, -1, -1):
            if not used[i]:
                used[i]=True
                cnt+=1
                if cnt==l:
                    break
    res=''
    for i in range(n):
        if used[i]:
            res+=S[i]
    return res

print(findmin('dagabaji', j))

kl=59*61
K=[False]*kl
L=[False]*kl
i=0
while 59*i+k<kl:
    K[59*i+k]=True
    i+=1
i=0
while 61*i+l<kl:
    L[61*i+l]=True
    i+=1
KL=0
# 最初に両者が共通する場所を探す
for i in range(kl):
    if K[i] and L[i]:
        KL=i
        break
# あとは59*61で循環してるはずなので、m回目にする
favorite1=kl*(m-1)+KL

# 完全数は少ないから列挙しておく
perfects=[6, 28, 496, 8128, 33550336]
for perf in perfects:
    num=abs(favorite1-perf)
    if num>=n:
        favorite2=perf
        break
else:
    raise Exception

print(min(favorite1, favorite2))
print(max(favorite1, favorite2))

print((o+p+q)*(r+s+t)*(u+v+w)*(x+y+z)%9973)
