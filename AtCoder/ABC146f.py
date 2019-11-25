# -*- coding: utf-8 -*-

"""
・600点自力AC！
・でも想定解はやっぱり後ろから貪欲じゃなかったみたい。
・貪欲も嘘解ではないみたいだから、結果的に問題はないんだけどね。。
・ただこういう問題って、強い人ほど貪欲しないからやっぱ引っかかるよね。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
S = input()[::-1]

# 地点aから行ける最大位置を返す
def check(a):
    for i in range(min(N, a+M), a, -1):
        if S[i] == '0':
            return i
    return a

i = 0
ans = []
while i < N:
    # 行けるところまで行く
    res = check(i)
    # 1歩も動けなければNG
    if res == i:
        print(-1)
        exit()
    ans.append(res-i)
    i = res
# 逆からやったので、もう1回逆順して出力
print(*ans[::-1])
