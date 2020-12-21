"""
参考：https://twitter.com/titia_til/status/1272936496333348864
・終了後挑むも自力ならず。。
・みんな既出よく思い出せるなぁ。コンテスト中に見ても多分ピンと来なかったな。
・AGC024Bと一緒。まじで座圧する以外そのまんまじゃん。。
・半年くらい前に自力ACしてるやつだったのに。。今見ても同じ発想が全く出なかった。
・誰かが呟いてたけど、LISみがあって騙されるはすげー分かる。
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

def compress(S):
    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

for _ in range(INT()):
    N = INT()
    A = LIST()

    # 座標圧縮
    zipped, _ = compress(A)
    for i in range(N):
        A[i] = zipped[A[i]]

    # 値が出現したindexを格納
    B = [0] * N
    for i, a in enumerate(A):
        B[a] = i

    # 連続した値が一番多く増加を続ける場所を見つける
    mx = cnt = 1
    for i in range(N-1):
        if B[i] < B[i+1]:
            cnt += 1
        else:
            cnt = 1
        mx = max(mx, cnt)
    print(N - mx)
