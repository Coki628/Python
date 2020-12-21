# -*- coding: utf-8 -*-

"""
・久しぶりにやってみたら時間かかったけど何とか自力AC
・とはいえ300点に時間かかりすぎた。。
・再帰で全探索
・「選ばない」っていう選択肢も含めれば全探索できる。
・あとは細かい例外処理。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def LIST2(n): return [INT() for i in range(n)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, A, B, C = MAP()
L = LIST2(N)

def calc(a, b, c):
    # 1本も選んでいない場所があるとNG(無からは作れないから)
    if a==0 or b==0 or c==0:
        return INF
    # 問題なければ増減のコストを返却
    return abs(A-a) + abs(B-b) + abs(C-c)

def rec(cur, a, b, c):
    if cur == N:
        return calc(a, b, c)
    return min(
        # 最初の1回は合成コストがかからない
        rec(cur+1, a+L[cur], b, c) + (10 if a!=0 else 0),
        rec(cur+1, a, b+L[cur], c) + (10 if b!=0 else 0),
        rec(cur+1, a, b, c+L[cur]) + (10 if c!=0 else 0),
        rec(cur+1, a, b, c),
    )

print(rec(0, 0, 0, 0))
