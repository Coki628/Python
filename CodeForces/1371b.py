"""
・自力ACここまで。
・等差数列の和
・いくつか試してみると、行ける範囲ではkひとつ変わる毎にできるパターンが1増えるようになると分かる。
・k < Nまではこの等差数列で、k == Nなら1つだけ作れて、k > Nは不可。これをr以下の範囲に対して考えるといい。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def get_sum(a, d, n):
    return (2*a + (n-1)*d) * n // 2

for _ in range(INT()):
    N, r = MAP()

    r = min(N, r)
    ans = 0
    if N == r:
        ans += 1
        r -= 1
    ans += get_sum(1, 1, r)
    print(ans)
