# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-thanks-festival-2017-open/editorial.pdf
・インタラクティブ問題
・N進数の各桁に割り当てることで、合計する前の各値が分かる。
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

# 10進数～N進数変換
def Base_10_to_n(X, n):
    if (int(X/n)):
        return Base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

N=INT()
coins=[5**i for i in range(4, -1, -1)]
ans=[0]*N
for i in range(10):
    Q=[0]*50
    for j in range(i*5, i*5+5):
        # 今回のクエリでの対象位置にコイン枚数を入れる
        Q[j]=coins[j%5]
    print('?', *Q[:N], flush=True)
    # コイン枚数の合計を保持
    sm=sum(Q[:N])
    res=INT()
    # 5進数の0～4と合わせるので、枚数*8を引く
    res-=sm*8
    # 5進数に変換して5桁で0埋め
    res=Base_10_to_n(res, 5).zfill(5)
    for j in range(i*5, i*5+5):
        if j>=N:
            # Nまで行ったら即終了
            print('!', *ans, flush=True)
            exit()
        # 0～4が8～12に対応しているので、ちょうど奇数の場所は一致してる
        if int(res[j%5])%2==1:
            # 9,11になる場所をTrue
            ans[j]=1
# Nが50まであればこっちで終了
print('!', *ans, flush=True)
