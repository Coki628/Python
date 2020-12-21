# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2018/11/25/153627
　　　https://qiita.com/kakira9618/items/dfd87d89f2c446bd7a78
　　　https://img.atcoder.jp/code-thanks-festival-2018/editorial.pdf
・辞書順最小
・条件を色々整えて貪欲に解く。
・何すればいいか理解するの大変だった…。
・pythonは1.6秒、pypyで0.8秒くらいAC
・計算量O(M*N^2)=2700万くらい
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
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
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M, K = MAP()
pN = LIST()

nodes = [[] for i in range(N)]
depths = [0] * N
# その時点で置けるノードかどうか
valid = [True] * N

root = -1
for i in range(N):
    if pN[i] != 0:
        # 隣接リスト
        nodes[pN[i]-1].append(i)
    else:
        root = i

# 各ノードの深さを取得
def upd_dpth(u, depth):
    depths[u] = depth
    for v in nodes[u]:
        upd_dpth(v, depth+1)

# コインを置けるノードかどうかを更新
def upd_valid(u):
    valid[u] = False
    for v in nodes[u]:
        upd_valid(v)

# 後m枚のコインを置くときに、できる操作の最大回数
def get_max(m):
    _nodes = []
    for i in range(N):
        # 有効なノードの深さを集める
        if valid[i]:
            _nodes.append(depths[i])
    # そもそも置きたい枚数のコインが置けないのでダメ
    if len(_nodes) < m:
        return -INF
    # 問題なければソートして大きいm個の合計操作回数(深さ+1)を返却
    _nodes.sort(reverse=True)
    res = 0
    for i in range(m):
        res += _nodes[i] + 1
    return res

# 後m枚のコインを置くときに、できる操作の最小回数
def get_min(m):
    _nodes = []
    for i in range(N):
        # 有効なノードの深さを集める
        if valid[i]:
            _nodes.append(depths[i])
    # そもそも置きたい枚数のコインが置けないのでダメ
    if len(_nodes) < m:
        return INF
    # 問題なければソートして小さいm個の合計操作回数(深さ+1)を返却
    _nodes.sort()
    res = 0
    for i in range(m):
        res += _nodes[i] + 1
    return res

# まず各ノードの深さは取得しておく
upd_dpth(root, 0)

ans = []
# コインの枚数ループ
for i in range(M):
    # 番号の若い頂点から試していく
    for j in range(N):
        # ノードjが置ける状態なら確認開始
        if valid[j]:
            # このjで操作をしない場合に戻す用
            history = copy(valid)
            # ノードj以下の部分木を使用不可にする
            upd_valid(j)
            # 今回の操作の後に、できる操作回数の最大と最小を取得
            mx = get_max(M - i - 1)
            mn = get_min(M - i - 1)
            # ノードjを置く場合にかかる操作回数
            opr = depths[j] + 1
            # 今回の操作を引いたKが範囲内かどうか
            if mn <= K - opr <= mx:
                # 1-indexed
                ans.append(j+1)
                K -= opr
                break
            # 今回のjはダメだったのでvalidの情報を復元
            valid = copy(history)
    else:
        # N頂点全て見ても置ける頂点がなかったのでNG
        print(-1)
        exit()
# 全てのコインを正しく置くことができた
print(*ans)
