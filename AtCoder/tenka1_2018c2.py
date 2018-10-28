# -*- coding: utf-8 -*-

# 終了後、重複処理を整理した版

from collections import deque

N = int(input())
aN = [int(input()) for i in range(N)]

# 受け取ったリストを前後から交互に並べていく
def changeList(_list):
    _list = deque(_list)
    res = [0] * N
    for i in range(N):
        if i % 2 == 0:
            res[i] = _list.popleft()
        else:
            res[i] = _list.pop()
    return res

# 受け取ったリストの隣り合う差の合計、最後だけ先頭とも比べて大きい方
def sumList(_list):
    ans = 0
    for i in range(1,N):
        if i == N-1:
            ans += max(abs(_list[i-1] - _list[i]), abs(_list[i] - _list[0]))
        else:
            ans += abs(_list[i-1] - _list[i])
    return ans

# 昇順と降順、両方試して良かった方を使う
aN2 = changeList(sorted(aN))
aN3 = changeList(sorted(aN, reverse=True))

print(max(sumList(aN2), sumList(aN3)))