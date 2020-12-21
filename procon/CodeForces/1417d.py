"""
参考：https://twitter.com/50j1_/status/1310266163490729985
　　　https://twitter.com/fuppy_kyopro/status/1310278568367153152
・自力ならず。。割と近い考え方までは行ってたけど詰めきれなかった。
・数列、構築
・まず均等に割り振れないならNG。で、OKなら1に全部集めて配り直す。ここまでは出た。
・配る時に足りない場合の扱いをうまく考えられなかった。
　方法は、まず集める時に余りが出て1以外のところに数が残ってしまうところを、
　1から先に、余りがなくなるまで配って全部ぴったり回収できるようにしてから集める、とやる。
・1から先に配ろうとすると、1に数が足りない時がないか心配になるけど、
　これは前から順番にやっていくことでうまくいく。
　(modが前からだんだん大きくなっていく操作なので、小さい方から全部集めながらやれば、
　足りなくなることがない。A[i]>=1 の条件がここで効いてくる。)
"""

import sys

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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()

    sm = sum(A)
    if sm % N:
        print(-1)
        continue

    each = sm // N
    ans = []
    for i in range(1, N):
        need = -A[i] % (i+1)
        A[i] += need
        A[0] -= need
        ans.append((1, i+1, need))
        x = A[i] // (i+1)
        A[i] -= (i+1)*x
        A[0] += (i+1)*x
        ans.append((i+1, 1, x))

    for i in range(1, N):
        x = each - A[i]
        A[i] += x
        A[0] -= x
        ans.append((1, i+1, x))

    print(len(ans))
    for a in ans:
        print(*a)
