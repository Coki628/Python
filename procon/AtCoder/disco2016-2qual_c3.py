# -*- coding: utf-8 -*-

"""
参考：http://ddcc2016-qual.contest.atcoder.jp/data/other/ddcc2016-qual/editorial.pdf
　　　http://procon-nenuon61.hatenablog.com/entry/2017/09/23/140802
　　　https://atcoder.jp/contests/ddcc2016-qual/submissions/4309742
・GCD毎に集計してまとめる。
・答えへの加算は同じキーと違うキーで扱いが変わるので場合分けする。
・dictをlistにして、重複なくペアのループを回せるようにした版
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
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

N,K=MAP()
A=LIST()

C=Counter()
for i in range(N):
    # KとのGCD毎に集計する
    C[gcd(A[i], K)]+=1

# 全ペアを効率よく回すためにdictをlistにしておく
L=list(C.items())
ans=0
# この2重ループならNより相当小さいので回せる
for i in range(len(L)):
    k1,v1=L[i]
    for j in range(i, len(L)):
        k2,v2=L[j]
        if k1*k2%K==0:
            # 同じキーなら、nC2の組み合わせ
            if k1==k2:
                ans+=v1*(v1-1)//2
            # 違うキーなら、全組み合わせ
            else:
                ans+=v1*v2
print(ans)
