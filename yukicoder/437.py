"""
・自力AC
・再帰探索
・最大文字数Nが13文字でだいぶ少ないのと、条件を満たす切り取り方は結構限定的だなーと思って
　再帰で全探索した。毎回文字列スライスがちょっと気になったけど、それも文字数少ないから
　まあ問題ないでしょう、って感じで。
・結果的に余裕のpythonAC0.07秒。再帰全探索の計算量はほんと読めないな。。
"""

import sys

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

S = input()

def rec(S):
    N = len(S)
    res = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if S[i] != '0' and S[i] != S[j] and S[j] == S[k]:
                    res = max(res, rec(S[:i] + S[i+1:j] + S[j+1:k] + S[k+1:]) + int(S[i]+S[j]+S[k]))
    return res

ans = rec(S)
print(ans)
