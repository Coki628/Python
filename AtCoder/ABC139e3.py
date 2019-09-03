# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=UWbGRhF3Ozw
　　　https://maspypy.com/atcoder-参加感想-2019-09-01abc-139
・人の参考にしつつ、色々試したけどやっぱりTLE。。
・リストやセットのコピーやるよりは、別関数にしてそこでやる方がまだ速いみたい。
"""

import sys

def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

N = INT()
A = [[]]
for i in range(N):
    A.append(LIST()[::-1])

def check(li, cnt):
    prev = cnt
    skip = [False] * (N+1)
    # 前回変更があった場所のみでループを回す
    nxtli = set()
    for i in li:
        if len(A[i]) and not skip[i]:
            a = A[i][-1]
            if len(A[a]) and not skip[a]:
                b = A[a][-1]
                if i == b:
                    A[i].pop()
                    A[a].pop()
                    # 変更があった場所をセットに入れておく
                    if len(A[i]):
                        nxtli.add(i)
                    if len(A[a]):
                        nxtli.add(a)
                    cnt -= 1
                    skip[i] = skip[a] = True
    if prev == cnt:
        print(-1)
        exit()
    return nxtli, cnt

cnt = N*(N-1)//2
li = set(range(1, N+1))
day = 0
while cnt > 0:
    li, cnt = check(li, cnt)
    day += 1
print(day)
