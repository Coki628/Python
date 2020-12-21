# -*- coding: utf-8 -*-

"""
・AGC300点自力AC
・逆から行けば貪欲なんとかなるんじゃないかと思ったら、なんとかなった。
・1つ前さえ気にすればいいから、結局2文字以上にはならない感じ。
・公式解がDPで、この貪欲は嘘解法疑惑もあるみたいだが。。。
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

S = input()
N = len(S)

cur = ''
prev = ''
cnt = 0
for i in range(N-1, -1, -1):
    cur = S[i] + cur
    # 前回確定分と比較する
    if cur != prev:
        # 一致してなければ分割できるのでそこで区切る
        cnt += 1
        prev = cur
        cur = ''
print(cnt)
