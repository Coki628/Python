# -*- coding: utf-8 -*-

"""
・反転数は、マージソートの過程でカウントできる。
"""

import sys
def input(): return sys.stdin.readline().strip()

N = int(input())
aN = list(map(int, input().split()))

def merge(l, m, r):
    # 入れ替えた回数
    cnt = 0

    L = aN[l:m]
    R = aN[m:r]
    L.append(float('inf'))
    R.append(float('inf'))
    i = j = 0
    for k in range(l, r):
        if L[i] <= R[j]:
            aN[k] = L[i]
            i += 1
        else:
            aN[k] = R[j]
            j += 1
            # 右のが先に入った時は入れ替わったってこと
            # カウントに追加するのは、Lの要素数 - 番兵 - Lから既に確定した数
            cnt += len(L) - 1 - i 
    return cnt
    
def merge_sort(l, r):
    # 入れ替えた回数    
    cnt = 0
    # l==rの時は何も変わらない
    if l+1 < r:
        # l==r(要素が1つ)になるまで再帰処理で分割する
        m = (l+r) // 2
        cnt += merge_sort(l, m)
        cnt += merge_sort(m, r)
        cnt += merge(l, m, r)
    # 左右から集めた分と今回のマージ分をまとめて返す
    return cnt

print(merge_sort(0, N))
