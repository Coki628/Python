"""
・自力AC！
・XORの性質、ORの性質
・サンプルがすごいヒントになった。
・とりあえず、3要素のXORで0になる条件があるので、各ビットはトータルで全部2回ずつ使うんだろうな、となる。
・あとはORの方だけど、これはサンプル見てたらピンと来てしまったんだけど、
　1つをNそのものにして、あとの2つにNを適当に振り分けると、どのペアもORはNになるし全ビット2回ずつになる。
・よって、Nが1ビットしか立ってない値だったらNG、
　それ以外なら適当に1ビットとその他全部みたいに分ければうまくいくのを作れる。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

N = INT()

for k in range(31):
    if N>>k & 1:
        a = 1<<k
        b = N ^ 1<<k
        break
c = N
if b:
    print(a, b, c)
else:
    print(-1, -1, -1)
