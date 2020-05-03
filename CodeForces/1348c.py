# -*- coding: utf-8 -*-

"""
・貪欲、鬼場合分け
・最大値の最小化でにぶたん？とか思ったけど対象が文字列だし違うよなぁ、
　ってなって、貪欲詰めるかぁってなったけど、なかなかいい方針が出なかった。
・最終的に、もしかしたらいけるかも、くらいの確度激低な貪欲投げて通ったけど、
　まじで大変だった。。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()
    S = list(input())

    # 1つなら自明に辞書順にするだけ
    if K == 1:
        S.sort()
        ans = ''.join(S)
        print(ans)
        continue

    S.sort(reverse=1)
    mx = S[0]
    A = [[] for i in range(K)]
    # とりあえず小さい方から1文字ずつ振る
    for i in range(K):
        A[i].append(S.pop())
    # この時点で最大より小さいのがあれば、そこに全振りしても最大に影響ないので、現時点の最大で確定
    if A[0] < A[-1]:
        ans = ''.join(A[-1])
    else:
        # この先に同値しか残ってないなら、均等に割り振る
        if S and S[-1] == mx:
            i = 0
            while S:
                A[i].append(S.pop())
                i = (i+1) % K
            ans = ''.join(A[0])
        # 大きいのがあったら、それを振った所が現時点の最大を抜かしてしまうので、現時点の最大に全振り
        else:
            A[-1] += sorted(S)
            ans = ''.join(A[-1])
    print(ans)
