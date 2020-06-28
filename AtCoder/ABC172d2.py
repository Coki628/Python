"""
参考：ツイートとか
・等差数列の和
・こっちのが公式解の方針だと思う。今回求める答えは約数の個数に*iするので、
　篩の内ループで毎回+jしたと考えると、その足される数は等差数列を作る。
・なのでNまでの項数を確認して等差数列の和を足せば内ループの分はO(1)で済む。
・頭いい。多分自分じゃ出ない。
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

N = INT()

ans = 0
for i in range(1, N+1):
    # 項数
    cnt = N // i
    # 等差数列の和
    ans += get_sum(i, i, cnt)
print(ans)
