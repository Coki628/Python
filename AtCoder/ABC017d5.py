# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc017
・1次元数え上げDP
・BITでやっていたところを累積和に。
・累積和で遷移を高速化
・区間和が欲しいのは自分の1つ手前までのどこかなので、
　自分の位置の値が分かった時点でその位置の累積和を更新しながら
　DPを進めればうまくいく。
・この解法だとpypyのがちょっと遅い。(まあでもどっちでも0.2秒くらい)
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

N,M=MAP()
A=[0]+[INT() for i in range(N)]

# dp[i] := サプリiまで摂取した状態の通り数
dp=[0]*(N+1)
dp[0]=1
acc=[0]*(N+2)
acc[1]=1
# サプリiが最後に出現した場所を保持
last=[0]*(M+1)
# 貰う遷移をどこまで遡ってやるかのidx
idx=0
for i in range(1, N+1):
    # サプリiが最後に出現した場所を取得(前回idxとmax取る)
    idx=max(idx, last[A[i]])
    # 上記場所を現在位置に更新
    last[A[i]]=i
    # idxから現在位置までの区間和を取得(まとめて遷移)
    sm=acc[i]-acc[idx]
    # dpの現在位置を更新
    dp[i]=sm%MOD
    # 累積和も更新
    acc[i+1]=(acc[i]+sm)%MOD
print(dp[N])
