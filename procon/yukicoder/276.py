"""
・なんとか自力AC
・等差数列の和、MODで考える。
・解法ガチャ回して回して、なんとか当てたみたいな感じではあったけど、
　一応自分なりに考察もしていたので書き記しておく。
・考えたのは、連続するN個の数は1つ進む毎に全体ではNずつ増えるので、常にmod Nでは一定だろうということ。
　そうなると、その等差数列の和がmod Nでいくつかが分かれば、そこで割り切ることは常にできるだろう、と。
・結果的にはそれでOKだった。実はNの偶奇でNかN/2のどっちかしかないって公式解説から分かったけど、
　まあ一応自分でちゃんと導き出せたからOKかな。
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

ans = get_sum(0, 1, N) % N
if ans == 0:
    print(N)
else:
    print(ans)
