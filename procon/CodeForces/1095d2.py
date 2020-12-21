# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/64130
　　　submission:47796520
・見てるのランダムな位置の隣り合う2人かと思ったら、
　自分の次と次の次の2人だった。
　こどふぉは問題文ちゃんと読むの大変…。
"""

import sys
input = sys.stdin.readline

N = int(input())
aN = [None] * N
for i in range(N):
    aN[i] = list(map(int, input().split()))
    # 0-indexed
    aN[i][0] -= 1
    aN[i][1] -= 1

nxt = 0
ans = [nxt]
# iは使わずnxtを見ていく
for _ in range(N-1):
    # もう一人の子を見ていた子が、次の子
    if aN[nxt][1] in aN[aN[nxt][0]]:
        nxt = aN[nxt][0]
    elif aN[nxt][0] in aN[aN[nxt][1]]:
        nxt = aN[nxt][1]
    ans.append(nxt)

for i in range(len(ans)):
    # 1-indexed
    ans[i] += 1
# 3人だけの時は両方隣り合っていてelifを見に行かないので
if N == 3:
    print(1, 2, 3)
else:
    print(*(ans))
