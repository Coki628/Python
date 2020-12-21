# -*- coding: utf-8 -*-

"""
・参考：http://arc060.contest.atcoder.jp/data/arc/060/editorial.pdf
　　　　https://www.youtube.com/watch?v=LzxPxJ-cSHk
・b <= √nは愚直に当たっていく。
・b > √nは、b^2 > nなので、b進数で2桁以下になる。
　例) 10^2は10進数で100、100 > n → nは99以下
　その2桁の数字はn = p*b + qと表せる。
　例) 67 = 6*10 + 7
　今回の問題の条件から、s = p + q
　n = p*b + q
　s = p + q
　上記2つの式から、連立方程式でqを消すと、
　n-s = p*b - p
　と表せる。
　pはb以上なら桁が上がってしまうので p < b
　p < b から p*p < p*b なので、p*b > p^2
　よって、n = p*b + q >= p*b > p^2
　n > p^2 なので、√n > p
　つまりpについて√nまでを全探索すればOK。
　n-s = p*b - p をbについての形に変形すればbが出せる。
　b = (n-s)/p + 1
・長かった…。
　こういう数学っぽい考察が要るやつ、ちゃんと出来るようになりたいなぁ。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import ceil, sqrt, hypot, factorial, pi, sin, cos, radians
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def f(b, n):
    if n < b:
        return n
    else:
        return f(b, n // b) + n % b

n = INT()
s = INT()

if s > n:
    print(-1)
elif s == n:
    print(n+1)
elif ceil(n/2) < s < n:
    print(-1)
else:
    # b <= √n の全探索
    for b in range(2, int(sqrt(n))+2):
        if f(b, n) == s:
            print(b)
            exit()
    # b > √n について、式を元に探す
    # pが大きい程bが小さいので、ループは逆順で回す
    for p in range(int(sqrt(n))+2, 0, -1):
        # 整数にならないものは除外
        if (n-s)%p: continue
        b = (n-s)//p + 1
        if f(b, n) == s:
            print(b)
            exit()
    # 見つからなかった
    print(-1)
