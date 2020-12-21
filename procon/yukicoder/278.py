"""
・さくっと自力AC
・数学考察で悩んだやつの上位置換だけど、その先は約数列挙やるだけ。
・青diffなのは多分多倍長ないとオーバーフローで値が陽に持てない分色々変形とか
　やらなきゃいけないからだと思う。pythonさんに感謝。。(ゆきここれ多いな)
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

def get_sum(a, d, n):
    """ 等差数列の和：(初項a, 公差d, 項数n) """
    return (2*a + (n-1)*d) * n // 2

def divisors(N: int) -> set:
    """ 約数の列挙・個数 """
    from math import sqrt

    s = {1, N}
    for i in range(2, int(sqrt(N))+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N = INT()

mx = get_sum(0, 1, N) % N
if mx == 0:
    mx = N
ans = sum(divisors(mx))
print(ans)
