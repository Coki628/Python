# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc074/editorial.pdf
・優先度付きキュー
・ダイクストラ以外で初めて使った。
・確かにこういう「詰めながらソートしたい」時には使えるんだな。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from heapq import heappush, heappushpop

N = int(input())
aN = list(map(int, input().split()))

mids = aN[N:N*2]
L = []
R = []
for i in range(N):
    # とりあえず前後半確定の部分をキューに詰める
    heappush(L, aN[i])
    heappush(R, -aN[i+N*2])

# 未確定の真ん中1/3について調べていく
l_sum = [0] * (N+1)
l_sum[0] = sum(aN[:N])
for i in range(N):
    # ある値をpushして最小値をpopする
    num = heappushpop(L, mids[i])
    # それに合わせて合計値も更新
    l_sum[i+1] = l_sum[i] + mids[i] - num

# 後半部分は逆方向からやる
r_sum = [0] * (N+1)
r_sum[N] = sum(aN[N*2:])
for i in range(N-1, -1, -1):
    # 正負逆転させれば最大値にも使える
    num = -(heappushpop(R, -mids[i]))
    r_sum[i] = r_sum[i+1] + mids[i] - num

ans = -float('inf')
for i in range(N+1):
    ans = max(l_sum[i] - r_sum[i], ans)
print(ans)
