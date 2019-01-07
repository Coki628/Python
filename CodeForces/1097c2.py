# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/64310
・最初の集計を想定解に寄せてみる(やってることはほとんど同じ)
"""

import sys
def input(): return sys.stdin.readline().strip()

from collections import deque, Counter

N = int(input())
aN = [input() for i in range(N)]

cnt = {'l': Counter(), 'r': Counter(), 'lr': 0}
for i in range(N):
    bracket = Counter()
    # 各行についてペア候補になる括弧を集計
    for j in range(len(aN[i])):
        if aN[i][j] == '(':
            bracket['l'] += 1
        else:
            if bracket['l'] != 0:
                bracket['l'] -= 1
            else:
                bracket['r'] += 1
    # どの状態の行がいくつあるか集計
    if bracket['l'] == 0 and bracket['r'] == 0:
        cnt['lr'] += 1
    elif bracket['l'] == 0:
        cnt['r'][bracket['r']] += 1
    elif bracket['r'] == 0:
        cnt['l'][bracket['l']] += 1

# 集計結果から、いくつペアが作れるか数える
ans = 0
for k, v in cnt['l'].items():
    ans += min(v, cnt['r'][k])
ans += cnt['lr'] // 2
print(ans)
