"""
・さすがにeval一撃だけでは何の成長もないのでやった。
・構文解析
・これは何とかなったけど、掛け割りもありになったらすげーややこしくなりそうだな。。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

S = input()
# 括弧始まりだとバグるので
if S[0] == '(':
    S = '0+' + S
N = len(S)

cur = 0
def rec(i, cur):
    cur += int(S[i])
    i += 1
    while i < N:
        if S[i] == '(':
            # 括弧を見つけたら、その中の計算を先に処理する
            nxti, tmp = rec(i+1, 0)
            if S[i-1] == '+':
                cur += tmp
            elif S[i-1] == '-':
                cur -= tmp
            i = nxti
        elif S[i] == ')':
            return i, cur
        elif S[i] in digits:
            if S[i-1] == '+':
                cur += int(S[i])
            elif S[i-1] == '-':
                cur -= int(S[i])
        i += 1
    return i, cur

_, ans = rec(0, 0)
print(ans)
