"""
・自力AC
・ゲーム、ソートして貪欲
・その場所を取った時の自分の利得的なものはa+bになるので、それでソートしてお互いに貪欲に取る。
"""

import sys
from typing import List

def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Solution:
    def stoneGameVI(self, A: List[int], B: List[int]) -> int:
        N = len(A)
        C = [0] * N
        for i in range(N):
            C[i] = (A[i]+B[i], A[i], B[i])
        C.sort(reverse=1)

        alice = 0
        bob = 0
        for i in range(N):
            if i % 2 == 0:
                alice += C[i][1]
            else:
                bob += C[i][2]
        if alice > bob:
            return 1
        elif alice < bob:
            return -1
        else:
            return 0
