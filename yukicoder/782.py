"""
・自力AC
・負のN進数
・あっとこで昔解いたやつ(ABC105C)を思い出した。
・そっちは-2進数限定だったけど、考え方はほぼ同じ感じでいける。
・てか、その時のコードを貼って、ちょっと加工して、AC。
・復習になっていいんだけど、昔のやつ引っ張り出してくるだけだと、
　本質として成長したのかはちょっと心配になる。。
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

def dton(num, n):
    res = []
    while abs(num) > 0:
        m = num % abs(n)
        num -= m
        res.append(m)
        num //= n
    res = ''.join(map(str, res))[::-1]
    if res:
        return res
    else:
        return '0'

T, B = MAP()
for _ in range(T):
    N = INT()
    ans = dton(N, B)
    print(ans)
