# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc101/editorial.pdf
　　　https://www.youtube.com/watch?v=5O4k1n5Ejzw
・二分探索、累積和、BIT(転倒数)
・さすがに700点は考察も実装も盛りだくさんだなぁ。
・計算量はBITのlogとにぶたんのlogでO(NlogNlogA)
・Aに重複なく10万要素あると、10万*17くらい*17くらいで3000万弱、pypyAC0.7秒
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

class BIT:
    def __init__(self, n):
        # 0-indexed
        self.size = n
        self.tree = [0] * (n+1)

    # [0, i]の区間和
    def sum(self, i):
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    # 区間和の取得 [l, r)
    def get(self, l, r=None):
        # 引数が1つなら一点の値を取得
        if r is None: r = l+1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

    # 値の追加：添字, 値
    def add(self, i, x):
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

N=INT()
A=LIST()
# 区間の全組み合わせ(累積和配列N+1から2つ選ぶ組み合わせ)
ALL=(N+1)*N//2

# 転倒数の取得
def get_inv(A):
    N=len(A)
    # マイナスの値を考慮するため今回は*2する
    bit=BIT(N*2+1)
    ans=0
    for i in range(N-1, -1, -1):
        bit.add(A[i]+N, 1)
        # 自分より右にある、自分より小さな数の個数
        ans+=bit.sum(A[i]+N-1)
    return ans

def calc(X):
    # X以上かどうかで1と-1にする配列L
    L=[0]*N
    for i in range(N):
        if A[i]>=X:
            L[i]=1
        else:
            L[i]=-1
    # 累積和
    sm=[0]+list(accumulate(L))
    # 区間の全組み合わせ - 累積和の転倒数(1より-1のが多かった区間の数)
    # = 中央値がX以上となる区間の数(1のが多いor同数だった区間の数)
    ans=ALL-get_inv(sm)
    # 中央値がX以上となる区間が、全体の半分(切り上げ)以上あるかどうか
    return ans>=ceil(ALL, 2)

# 二分探索
A2=sorted(set(A))
low=0
hi=len(A2)
while low+1<hi:
    mid=(hi+low)//2
    if calc(A2[mid]):
        low=mid
    else:
        hi=mid
# 今回は境界の下が欲しいのでlowを出力
print(A2[low])
