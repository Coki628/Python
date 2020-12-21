"""
・自力AC
・間違ってる演算子の位置は「符号符号数字」の左か「数字符号数字」の真ん中のどちらか。
・そこさえ正しく修正すれば後はevalでよしなに計算できる。
"""

import sys
from string import digits

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

S = list(input())
N = len(S)

for i in range(N-2):
    if S[i] not in digits and S[i+1] not in digits and S[i+2] in digits:
        if S[i] == '+':
            S[i] = '-'
        else:
            S[i] = '+'
    if S[i] in digits and S[i+1] not in digits and S[i+2] in digits:
        if S[i+1] == '+':
            S[i+1] = '-'
        else:
            S[i+1] = '+'

ans = eval(''.join(S))
print(ans)
