# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/64310
・最初の集計を想定解に寄せてみる(やってることはほとんど同じ)
"""

import sys
def input(): return sys.stdin.readline().strip()

from collections import Counter

N = int(input())

cnt = {'l': Counter(), 'r': Counter(), 'lr': 0}
for _ in range(N):
    l = r = 0
    # 各行についてペア候補になる括弧を集計
    for c in input():
        if c == '(':
            l += 1
        else:
            if l != 0:
                l -= 1
            else:
                r += 1
    # どの状態の行がいくつあるか集計
    if l == 0 and r == 0:
        cnt['lr'] += 1
    elif l == 0:
        cnt['r'][r] += 1
    elif r == 0:
        cnt['l'][l] += 1

# 集計結果から、いくつペアが作れるか数える
ans = 0
for k, v in cnt['l'].items():
    ans += min(v, cnt['r'][k])
ans += cnt['lr'] // 2
print(ans)
