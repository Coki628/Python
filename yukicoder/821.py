"""
・結構さくっと自力AC！
・まず、数列中のどの値に-1をかけるかを考えて、その時に出来上がる和を考えると、
　途中で相当な数の重複が発生するのが分かる。このままではnCrとかで通り数をうまく数えたりできない。
・で、もうしばらく観察すると、減らせる値は重複はしているが、
　間に隙間が空くことなく最後まで全ての値を網羅している、となる。
・すると最大の場合だけに注目すれば、その間は全て網羅できるので考えなくていい、となる。
　(この考察、どっかのABCのDかEでもやった)
・これで最大に減らせる場合を見つけるだけの作業になるので、
　数列Aの大きい方からK個を取る、をやればよく、これはsum(A[N-K+1:])なので、
　数列Aが1ずつ増えることから、公差1の等差数列をO(1)で出せばいい。
　これに、何も減らさない場合の+1をすれば答え。
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

N, K = MAP()

ans = get_sum(N-K+1, 1, K) + 1
print(ans)
