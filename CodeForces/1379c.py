"""
参考：https://twitter.com/laycrs/status/1284806393417981953
・自力ならず。
・累積和、二分探索
・2個目以上使うのは1種類で良い、それを全探索。
　まずこの考察が出なかったので、スタートラインにも立ててなかった。。
・あとは基本は累積和にぶたんなんだけど、1個目を買うところの累積和部分に、
　自分iが含まれるかどうかで場合分けが必要。これの実装詰めるのに時間かかってしまった。
・考察出ないのはしょうがないけど、せめて実装パートは爆速で仕上げたいもんだ。。
"""

import sys
from itertools import accumulate
from bisect import bisect_right

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

T = INT()
for t in range(T):
    N, M = MAP()
    A = [0] * M
    B = [0] * M
    for i in range(M):
        A[i], B[i] = MAP()

    A2 = sorted(A)
    acc = list(accumulate(A2[::-1]))[::-1] + [0]
    A2.append(INF)
    ans = 0
    for i in range(M):
        # 自分のBより大きいところまでAを取る
        j = bisect_right(A2, B[i])
        cur = 0
        cnt = M - j
        # Nが小さいとcnt個取れないのでN個まで少なくする
        if cnt > N:
            j = M - N
            cnt = N
        # 取るAの中に自分A[i]が含まれていなければ含める
        if A2[j] > A[i]:
            cur += A[i]
            # 自分のAを含めた時に、Bから取る分が0個でそこで調節できない時は、累積和部分から減らす
            if cnt == N:
                j += 1
            else:
                cnt += 1
        cur += acc[j] + (N-cnt)*B[i]
        ans = max(ans, cur)
    print(ans)

    if t != T-1:
        input()
