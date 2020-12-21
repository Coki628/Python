"""
・三角形の性質
・これほんと競プロ頻出だな。
・今回のは制約小さいので、ソートしたら脳死で3重ループ回せばOK。
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
MOD = 10 ** 19 + 7
EPS = 10 ** -10

N = INT()
A = LIST()

A.sort()
ans = 0
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            a, b, c = A[i], A[j], A[k]
            if a == b or b == c or c == a:
                continue
            # 小さい2辺を合わせて一番長い辺より長くなれば三角形が作れる
            if a+b > c:
                ans += 1
print(ans)
