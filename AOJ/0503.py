# -*- coding: utf-8 -*-

"""
・蟻本演習2-1-3、ゲームでBFS
・状態数が多すぎるのか、MLEした。。
"""

import sys
from collections import defaultdict, deque

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def deepcopy(li): return [x[:] for x in li]

def bfs(tray):

    memo = defaultdict(lambda: defaultdict(set))
    que = deque()
    que.append((tray, 0))
    res = INF

    while len(que):

        tray, cost = que.popleft()
        # 真ん中が空なら成功、手数を記録する
        if len(tray[1]) == N or len(tray[3]) == N:
            res = min(res, cost)
            continue
        key1, key2, key3 = tuple(tray[1]), tuple(tray[2]), tuple(tray[3])
        if key3 in memo[key1][key2]:
            continue
        memo[key1][key2].add(key3)

        for i in range(1, 4):
            if len(tray[i]):
                # 隣が空か自分より小さければ、コップを移動できる
                if not len(tray[i-1]) or tray[i-1][-1] < tray[i][-1]:
                    nxt = deepcopy(tray)
                    nxt[i-1].append(nxt[i].pop())
                    que.append((nxt, cost+1))
                if not len(tray[i+1]) or tray[i+1][-1] < tray[i][-1]:
                    nxt = deepcopy(tray)
                    nxt[i+1].append(nxt[i].pop())
                    que.append((nxt, cost+1))
    # 動ける手がなくなったら最小手数を返す
    return res

while True:
    N, M = MAP()
    if N == M == 0:
        break
    tray = []
    # 左右端に番兵を仕込んでおく
    tray.append([INF])
    tray.append(LIST()[1:])
    tray.append(LIST()[1:])
    tray.append(LIST()[1:])
    tray.append([INF])
    
    res = bfs(tray)
    if res <= M:
        print(res)
    else:
        print(-1)
