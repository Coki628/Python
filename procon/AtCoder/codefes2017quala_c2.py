# -*- coding: utf-8 -*-

"""
・400点自力AC！
・回文行列
・似たようなのCFでやったことあってよかった。(1118c)
・ちょっと整理した。
"""

import sys
from collections import Counter

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

H,W=MAP()
S=''
for i in range(H):
    S+=input()

# 各文字を4つ組と余りにまとめる
C=Counter(S)
cnt=Counter()
for v in C.values():
    cnt[v%4]+=1
# 3つ組は1と2に割り振る
cnt[1]+=cnt[3]
cnt[2]+=cnt[3]
cnt[3]=0

# 縦横とも偶数
if H%2==0 and W%2==0:
    # 全て4つ組
    if cnt[1]==0 and cnt[2]==0:
        Yes()
    else:
        No()

# 縦偶数
elif H%2==0:
    # 2つ組がH/2以下で残りが4つ組
    if cnt[2]<=H//2 and cnt[1]==0:
        Yes()
    else:
        No()

# 横偶数
elif W%2==0:
    # 2つ組がW/2以下で残りが4つ組
    if cnt[2]<=W//2 and cnt[1]==0:
        Yes()
    else:
        No()

# 両方奇数
else:
    # 1つ組1つ、2つ組が(H+W)/2-1以下で残りが4つ組
    if cnt[1]==1 and cnt[2]<=(H+W)//2-1:
        Yes()
    else:
        No()
