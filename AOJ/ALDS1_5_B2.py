# -*- coding: utf-8 -*-

"""
マージソート
・再帰で全部分割してから統合
・部分配列の切り出しを、for文回さないでスライスにした。TLE→AC
"""

import sys
def input(): return sys.stdin.readline().strip()

N = int(input())
aN = list(map(int, input().split()))
ans = 0

def merge(l, m, r):
    # 比較回数
    global ans    

    # aNからl～mとm～rの範囲を取り出す
    L = aN[l:m]
    R = aN[m:r]
    # 番兵(最後の要素の処理をしやすくする)
    L.append(float('inf'))
    R.append(float('inf'))

    i = j = 0
    # 前から順番にLとRを比べて、
    # 小さい方からaNに更新かけて添字を進める
    for k in range(l, r):
        # ここが=付でL優先するから安定ソート
        if L[i] <= R[j]:
            aN[k] = L[i]
            i += 1
        else:
            aN[k] = R[j]
            j += 1
        ans += 1

def merge_sort(l, r):
    # l==rの時は何もしない
    if l+1 < r:
        # l==r(要素が1つ)になるまで再帰処理で分割する
        m = (l+r) // 2
        merge_sort(l, m)
        merge_sort(m, r)
        # 統合していく
        merge(l, m, r)

merge_sort(0, N)
print(*aN)
print(ans)
