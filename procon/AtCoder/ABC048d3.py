# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc064/editorial.pdf
・最終的には両端にある2種類の文字か、両端が同じ文字ならそれとそれ以外のもう1文字以外は残らない。
・なので残った文字数の偶奇は分かる。
・そこから総ターン数が分かるので、偶奇で勝敗が分かる。
"""

import sys
from collections import deque, Counter, defaultdict
from math import sqrt, hypot, factorial, pi, sin, cos, radians
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, product, combinations, combinations_with_replacement
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial

def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

S = input()
N = len(S)

# この条件はxorを使うとスッキリまとまる
if (S[0] == S[-1]) ^ (N % 2 == 0):
    print('Second')
else:
    print('First')
