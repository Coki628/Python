# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc009
　　　https://atarimae.biz/archives/23930
　　　http://tutuz.hateblo.jp/entry/2019/01/29/212325
　　　http://kamui108.hatenablog.com/entry/2016/08/17/004147
　　　https://atcoder.jp/contests/abc009/submissions/5421410
・行列の積、行列累乗、繰り返し二乗法
・漸化式は、初期値*行列累乗の形に落とし込めることがある。
・半環といって、XORを足し算、ANDを掛け算として扱って行列の計算が出来る。
・遷移に使う行列(累乗するやつ)は頑張って導き出す必要がある。
・掛け算は単位元が1だけど、ANDの単位元はビットの全部埋めだから気を付ける。
・繰り返し二乗法はダブリングの簡易版みたいな感じ。
・計算量は行列積のK^3と繰り返し二乗法のlogMで100万*30くらいの3000万、pypyで0.7秒AC。
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

K,M=MAP()
A=LIST()
C=LIST()

def mat_dot(A, B):
    """ 行列の積(*がAND,+がXOR版) """

    # 1次元リストが来たら2次元の行列にする
    if not isinstance(A[0], list):
        A=[A]
    if not isinstance(B[0], list):
        B=[[b] for b in B]
    n1=len(A)
    n2=len(A[0])
    m1=len(B)
    m2=len(B[0])
    res=list2d(n1, m2, 0)
    for i in range(n1):
        for j in range(m2):
            for k in range(n2):
                res[i][j]^=A[i][k]&B[k][j]
    return res

# 例外処理
if M<K:
    print(A[M-1])
    exit()

# ANDの単位元はbit全埋め
init=(1<<32)-1
# 遷移に使う行列
C2=list2d(K, K, 0)
C2[0]=C
for i in range(1, K):
    C2[i][i-1]=init

N=M-K
C3=list2d(K, K, 0)
# 斜めに単位元を入れる、この形にすれば何が来ても値を変えない
for i in range(K):
    C3[i][i]=init

nxt=C2
for i in range(31):
    if N>>i&1:
        C3=mat_dot(C3, nxt)
    # 繰り返し二乗法
    nxt=mat_dot(nxt, nxt)
# 最後に累乗した遷移行列と最初の値をかける
ans=mat_dot(C3, A[::-1])
print(ans[0][0])
