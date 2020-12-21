# -*- coding: utf-8 -*-

"""
参考：http://code-festival-2016-quala.contest.atcoder.jp/data/other/code-festival-2016-qualb/editorial.pdf
・公式解
・最小全域木
・各行列はまとめて考えていい。
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

W,H=MAP()
P=[(INT(), 'p') for i in range(W)]
Q=[(INT(), 'q') for i in range(H)]

# 行列をまとめて、コストが低い方から進める
PQ=P+Q
PQ.sort()

a=H+1
b=W+1
ans=0
for i in range(W+H):
    val,_type=PQ[i]
    if _type=='p':
        # 横線を行数に応じて足して、列数を1つ減らす
        ans+=a*val
        b-=1
    else:
        # 縦線を列数に応じて足して、行数を1つ減らす
        ans+=b*val
        a-=1
print(ans)
