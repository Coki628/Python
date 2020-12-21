"""
・ここまで自力AC。でもこれはもうちょい手早くやれたな。。
・ソート、実装
・予定数の少ない人から順に最初の1つは大きい値から配る。これで最大値は最適。
・残りは予定数の多い人から小さい値を渡せるだけ渡す。
　これで小さい数はなるべく1人に寄せれるので最小値も最適。
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

for _ in range(INT()):
    N, K = MAP()
    A = LIST()
    B = LIST()

    # 最初の1つを大きい方から渡す
    A.sort(reverse=1)
    B.sort()
    j = 0
    adjli = [[] for i in range(K)]
    k = 0
    for i in range(K):
        adjli[i].append(A[i])
        B[i] -= 1

    # 残りは小さい方からなるべく1人に寄せる
    A.reverse()
    B.reverse()
    adjli.reverse()
    j = 0
    for i in range(N-K):
        while B[j] == 0:
            j += 1
        adjli[j].append(A[i])
        B[j] -= 1

    ans = 0
    for i in range(K):
        ans += min(adjli[i]) + max(adjli[i])
    print(ans)
