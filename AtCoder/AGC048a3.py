"""
・より条件をシンプルにした。これでも通った。
・まあ本質は最初の例外処理だよね。。
　よく考えたら、辞書順だと文字数が多ければaaaaaとかでも後ろになるもんね。。
"""

import sys
from string import ascii_lowercase

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    S = input()
    N = len(S)

    if S > 'atcoder':
        print(0)
        continue

    ans = -1
    T = ascii_lowercase[1:]
    for i in range(N):
        if S[i] in T:
            ans = i
            break
    T = 'uvwxyz'
    for i in range(1, N):
        if S[i] in T:
            ans = min(ans, i - 1)
            break
    print(ans)
