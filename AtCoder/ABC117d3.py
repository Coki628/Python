# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2019/02/04/013700
・XOR系
・桁DP
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

N,K=MAP()
A=LIST()

mx=max(K, max(A))
bmx=format(mx, 'b')
mxlen=len(bmx)
bK=format(K, '0'+str(mxlen)+'b')

# ビットが立っている数を各桁で集計しておく
cnt1=[0]*mxlen
for i in range(N):
    ba = format(A[i], '0'+str(mxlen)+'b')
    for j, bit in enumerate(ba):
        if bit=='1':
            cnt1[j]+=1
num0=[0]*mxlen
num1=[0]*mxlen
for i in range(mxlen):
    # カウントした方と逆になる
    num0[i]=cnt1[i]*2**(mxlen-i-1)
    num1[i]=(N-cnt1[i])*2**(mxlen-i-1)

# dp[i][j] := i桁目まで考えた時、j=0はi桁目まで一致(K=ans)、j=1は既にansがK未満(K>ans)になった状態
dp=list2d(mxlen+1, 2, -1)
dp[0][0]=0
for i in range(mxlen):
    if dp[i][1]!=-1:
        # smaller -> smaller
        dp[i+1][1]=max(dp[i+1][1], dp[i][1]+max(num0[i], num1[i]))
    if dp[i][0]!=-1:
        # exact -> exact
        if bK[i]=='0':
            dp[i+1][0]=max(dp[i+1][0], dp[i][0]+num0[i])
        else:
            dp[i+1][0]=max(dp[i+1][0], dp[i][0]+num1[i])
        # exact -> smaller
        if bK[i]=='1':
            dp[i+1][1]=max(dp[i+1][1], dp[i][0]+num0[i])
print(max(dp[mxlen][0], dp[mxlen][1]))
