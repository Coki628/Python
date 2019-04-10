# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/yahoo-procon2017-qual/editorial.pdf
　　　https://atcoder.jp/contests/yahoo-procon2017-qual/submissions/4355262
・公式解
"""

import sys
from os.path import commonprefix

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
A=[a-1 for a in A]
S=[input() for i in range(N)]

# 入れたい候補の共通接頭辞
S2=[S[i] for i in A]
cp=commonprefix(S2)
m=len(cp)

Aset=set(A)
# -1開始にすると、入れたくない候補0の例外処理を場合分けしなくて済む
M=-1
for i, s in enumerate(S):
    # 入れたい候補に含まれないものだけチェック
    if i not in Aset:
        # 入れたい候補の共通接頭辞と比較して最大を取る
        M=max(M, len(commonprefix([cp, s])))

if M<m:
    print(cp[:M+1])
else:
    print(-1)
