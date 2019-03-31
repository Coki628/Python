# -*- coding: utf-8 -*-

# 各種インポート
import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
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

# numpy系
import numpy as np
from scipy.sparse.csgraph import dijkstra, floyd_warshall

# 調査用
# import matplotlib.pyplot as plt 
# import pandas as pd

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)

# 小数点以下9桁まで表示(これやんないと自動でeとか使われる)
'{:.9f}'.format(3.1415)

# 文字列リバース
s = ''
s = s[::-1]

# int -> bin(str)
num = 1234
num = format(num, 'b')
# int -> bin(str) (8桁0埋め)
num = format(aN[i], '08b')
# bin(str) => int
num = int(num, 2)

# 二番目の要素でソート
aN = [[1, 2], [3, 1]]
aN.sort(key=lambda x: x[1])
# こっちのがちょっと速い
aN.sort(key=itemgetter(1))

# 四捨五入で整数に丸める
def round(x): return int((x*2+1) // 2)

# modの除算(フェルマーの小定理)
def fermat(x, y, MOD):
    return x * pow(y, MOD-2, MOD) % MOD

# 配列要素全部掛け(総乗)
prod = partial(reduce, mul)
# これでもよさげ
def prod(nums): return reduce(mul, nums, 1)
prod([1, 2, 3])
np.prod([1, 2, 3])

# 右左上下
directions = [(0,1),(0,-1),(1,0),(-1,0)]
# 四方に一回り大きいグリッドを作る
# grid = list2d(H+2, W+2, '*')
# for i in range(1,H+1):
#     row = list(input())
#     for j in range(1, W+1):
#         grid[i][j] = row[j-1]

# 余りの切り上げ(3つとも同じ)
# def ceil(a, b):
#     (a + b - 1) // b
#     (a - 1) // b + 1
#     return -(-a // b)

# 最小公倍数
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums):
    # reduce(使う関数, 足し合わせるリスト, 初期値)
    return reduce(lcm, nums, initial=1)

# 1からnまでの等差数列の和
def get_sum(n):
    return (1+n)*n//2

# 素数判定
def is_prime(num):
    if num < 2: 
        return False
    if num in [2, 3, 5]: 
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

# 素数列挙(エラトステネスの篩)
def eratosthenes_sieve(n):
    table = [0] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    return prime_list

# 素因数分解
def factorize(num: int) -> dict:
    d = Counter()
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

# 約数の列挙・個数
def divisor_set(N: int) -> set:
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て+1
    end = int(sqrt(N)) + 1
    for i in range(2, end+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

# 階乗たくさん使う時用のテーブル準備
# MAX：階乗に使う数値の最大以上まで作る
def init_fact_inv(MAX: int, MOD: int) -> list:
    # 階乗テーブル
    factorial = [1] * (MAX)
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i % MOD
    # 逆元テーブル
    inverse = [1] * (MAX)
    # powに第三引数入れると冪乗のmod付計算を高速にやってくれる
    inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
    for i in range(MAX-2, 0, -1):
        # 最後から戻っていくこのループならMAX回powするより処理が速い
        inverse[i] = inverse[i+1] * (i+1) % MOD
    return factorial, inverse

# 組み合わせの数(必要な階乗と逆元のテーブルを事前に作っておく)
def nCr(n, r):
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % MOD
    return numerator * denominator % MOD

# テーブル準備MODなし版
def init_factorial(MAX: int) -> list:
    # 階乗テーブル
    factorial = [1] * (MAX)
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i
    return factorial

# 組み合わせの数(必要な階乗のテーブルを事前に作っておく)
def nCr(n, r):
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = factorial[r] * factorial[n-r]
    return numerator // denominator

# テーブル準備logでやる版
def init_fact_log(MAX: int) -> list:
    fact_log = [0] * (MAX)
    for i in range(1, MAX):
        fact_log[i] = fact_log[i-1] + log10(i)
    return fact_log

def nCr(n, r):
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return round(pow(10, fact_log[n] - fact_log[r] - fact_log[n-r]))

# 事前テーブルなし組み合わせ簡易版
def nCr(n, r):
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

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

# ベルマンフォード(頂点数, 辺集合(0-indexed), 始点)
def bellman_ford(N: int, edges: list, src: int) -> list:
    # 頂点[ある始点からの最短距離]
    # (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [float('inf')] * N
    res[src] = 0
    # 各辺によるコストの置き換えを頂点数N-1回繰り返す
    for i in range(N-1):
        for src, dest, cost in edges:
            if res[dest] > res[src] + cost:
                res[dest] = res[src] + cost
    # 無限に減らせる場所がないか確認
    for src, dest, cost in edges:
        if res[dest] > res[src] + cost:
            # あったら空リストを返却
            return []
    # 問題なければ頂点リストを返却
    return res

# ワーシャルフロイド(頂点数, 隣接行列(0-indexed))
def warshall_floyd(N: int, graph: list) -> list:
    res = deepcopy(graph)
    # 全頂点の最短距離
    for k in range(N):
        for i in range(N):
            for j in range(N):
                # 始点 = 終点、は例外的に距離0にしておく
                if i == j:
                    res[i][j] = 0
                else:
                    res[i][j] = min(res[i][j], res[i][k] + res[k][j])
    return res

# トポロジカルソート(頂点数、辺集合(DAG, 0-indexed))
def topological_sort(N: int, edges: list) -> list:
    # ここからトポロジカルソート準備
    incnts = [0] * N
    outnodes = [[] for i in range(N)]
    for i in range(len(edges)):
        # 流入するノード数
        incnts[edges[i][1]] += 1
        # 流出先ノードのリスト
        outnodes[edges[i][0]].append(edges[i][1])
    # 流入ノード数が0であるノードのセットS
    S = set()
    for i in range(N):
        if incnts[i] == 0:
            S.add(i)

    # ここからトポロジカルソート
    L = []
    # 暫定セットが空になるまでループ
    while S:
        # 暫定セットから結果リストへ1つ入れる
        L.append(S.pop())
        # 確定させたノードから流出するノードでループ
        for node in outnodes[L[-1]]:
            # 流入ノード数を1減らす
            incnts[node] -= 1
            # 流入ノードが0なら暫定セットへ
            if incnts[node] == 0:
                S.add(node)
    # ソートされた頂点のリストを返却
    return L


# Union-Find木
class UnionFind:
    def __init__(self, n):
        # 親要素のノード番号を格納。par[x] == xの時そのノードは根
        # 1-indexedのままでOK、その場合は[0]は未使用
        self.par = [i for i in range(n+1)]
        # 木の高さを格納する（初期状態では0）
        self.rank = [0] * (n+1)
        # あるノードを根とする集合に属するノード数
        self.size = [1] * (n+1)

    # 根の検索(グループ番号と言えなくもない)
    def find(self, x):
        # 根ならその番号を返す
        if self.par[x] == x:
            return x
        else:
            # 走査していく過程で親を書き換える
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    # 併合
    def union(self, x, y):
        # 根を探す
        x = self.find(x)
        y = self.find(y)
        # 木の高さを比較し、低いほうから高いほうに辺を張る
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
            # 木の高さが同じなら片方を1増やす
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    # 同じ集合に属するか判定
    def same(self, x, y):
        return self.find(x) == self.find(y)


# 重み付きUnion-Find木
class WeightedUnionFind:
    def __init__(self, n):
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        # 根への距離を管理
        self.weight = [0] * (n+1)

    # 検索
    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            y = self.find(self.par[x])
            # 親への重みを追加しながら根まで走査
            self.weight[x] += self.weight[self.par[x]]
            self.par[x] = y
            return y

    # 併合
    def union(self, x, y, w):
        rx = self.find(x)
        ry = self.find(y)
        # xの木の高さ < yの木の高さ
        if self.rank[rx] < self.rank[ry]:
            self.par[rx] = ry
            self.weight[rx] = w - self.weight[x] + self.weight[y]
        # xの木の高さ ≧ yの木の高さ
        else:
            self.par[ry] = rx
            self.weight[ry] = - w - self.weight[y] + self.weight[x]
            # 木の高さが同じだった場合の処理
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    # 同じ集合に属するか
    def same(self, x, y):
        return self.find(x) == self.find(y)

    # xからyへのコスト
    def diff(self, x, y):
        return self.weight[x] - self.weight[y]
