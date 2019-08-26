# -*- coding: utf-8 -*-

"""
・今日の一問から
・なんとか自力AC、これまじで300点か。。
・結構印象に残ってた問題だったのもあって、解法の方針は思い出した。
　にも関わらず、高さ0位置の扱いでコーナーケースに苦しむ。
　結局、もう1回全部回って再確認って手段に出たが、なんとかAC取れてよかった。
・↑解法見たら、これこそ想定解で問題なかった。
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

N = INT()
XYH = []
for i in range(N):
    x, y, h = MAP()
    XYH.append((x, y, h))

# 中心候補として全座標を試す
for y1 in range(101):
    for x1 in range(101):
        H = set()
        for x2, y2, h in XYH:
            # 各座標情報から、中心の高さ(仮)を出す
            tmph = abs(x1-x2) + abs(y1-y2) + h
            # 高さ0の情報は使えないので除く
            if h > 0:
                H.add(tmph)
        # 合ってるか再確認
        if len(H) == 1:
            ansh = list(H)[0]
            for x2, y2, h in XYH:
                tmph =  ansh - (abs(x1-x2) + abs(y1-y2))
                if tmph < 0:
                    continue
                if tmph != h:
                    break
            else:
                print(x1, y1, list(H)[0])
                exit()
