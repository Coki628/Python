# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/soundhound2018-summer-qual/editorial.pdf
　　　https://ikatakos.com/pot/programming_algorithm/contest_history/atcoder/2018/0707_soundhound2018_summer_qual
・これは自力で解きたかった。
・制約的にダイクストラくらいしかないって分かってたけど、
　始点からと終点からのダイクストラ2回だけで全経由地への最短を網羅できるって気付けなかった…。
・まあ最後に逆順で回せばいいってのも、気付けたか分からんのだけどね…。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, xor
from copy import copy, deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
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

# ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点)
def dijkstra(N: int, nodes: list, src: int) -> list:
    # 頂点[ある始点からの最短距離]
    # (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [float('inf')] * N
    # スタート位置
    que = [(0, src)]
    res[src] = 0
    # キューが空になるまで
    while len(que) != 0:
        # srcからの距離, 現在のノード
        dist, cur = heappop(que)
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if res[cur] + cost < res[nxt]:
                res[nxt] = res[cur] + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (res[nxt], nxt))
    # ノードsrcからの最短距離リストを返却
    return res

N,M,S,T=MAP()
nodes1=[[] for i in range(N)]
nodes2=[[] for i in range(N)]
for i in range(M):
    u,v,a,b=MAP()
    # 円での始点からダイクストラ用
    nodes1[u-1].append((v-1,a))
    nodes1[v-1].append((u-1,a))
    # スヌークでの終点からダイクストラ用
    nodes2[u-1].append((v-1,b))
    nodes2[v-1].append((u-1,b))

yen=dijkstra(N,nodes1,S-1)
snuk=dijkstra(N,nodes2,T-1)

ans=[0]*N
mn=INF
# 逆順で最小値を取っていけば効率よくやれる
for i in range(N-1, -1, -1):
    ans[i]=mn=min(mn, yen[i]+snuk[i])
# 10^15から引けば答え
for i in range(N):
    print(1000000000000000-ans[i])
