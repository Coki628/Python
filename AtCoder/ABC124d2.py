# -*- coding: utf-8 -*-

"""
・400点なんとか自力AC
・累積和
・コンテスト後、ちょっと整理した版
"""

import sys
from itertools import accumulate

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

N,K=MAP()
S=input()

L=[]
cnt0=0
# 0始まりの時は番兵的に1の区間0を追加しておく
if S[0]=='0':
    L.append(0)
    cur=0
else:
    cur=1
cnt=0
# 1の区間の長さ、0の区間の長さ、1の区間の長さ...の配列Lを作る
for i in range(N):
    if cur==int(S[i]):
        cnt+=1
    else:
        L.append(cnt)
        if cur==0:
            cnt0+=1
        cnt=1
        # フラグの切り替え
        cur=1-cur
else:
    if cnt!=0:
        L.append(cnt)
        if cur==0:
            cnt0+=1
# 0終わりの時は番兵的に1の区間0を追加しておく
if S[-1]=='0':
    L.append(0)

acc=[0]+list(accumulate(L))
mx=0
if cnt0<K:
    # 0区間の数がKより少なかったら、それに合わせて区間和取る長さを短くする
    K=cnt0
for i in range(0, len(acc), 2):
    if i+(K*2+1)<len(acc):
        # 各区間和から最大を取る
        mx=max(mx, acc[i+(K*2+1)]-acc[i])
print(mx)
