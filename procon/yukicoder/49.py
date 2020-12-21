"""
・さくっと自力AC
・構文解析
・eval便利。
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
N = len(S)
# とりあえず*と+を普通にする
S = S.replace('*', '#').replace('+', '*').replace('#', '+')

ans = ''
i = 0
# 最初だけ(数字)(演算子)の形にする
while i < N:
    if S[i] in '+*':
        ans += S[i]
        i += 1
        break
    ans += S[i]
    i += 1
while i < N:
    if S[i] in '+*':
        # (数字)(演算子)(数字)になったところで計算
        ans = str(eval(ans))
    ans += S[i]
    i += 1
ans = str(eval(ans))
print(ans)
