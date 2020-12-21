# -*- coding: utf-8 -*-

"""
・色々ツイート参考でAC。自分ではどう貪欲すれば正解か見抜けなかった。
・結果的に、上からでも下からでもいけるらしい。自分は下からやった。
・分解にはコストがかかるけど、併合にはかからないので、今のビットで
　使い終わった余りは全部1つ上のビットに併合する。
・それでも足りない時は上に直近の立っているからビットから分解してくる。
　この時、上ビットを見に行くループ、下ビットに分解してくるループ、
　と愚直に処理してネストが深くなっても、高々ビット数くらいしか回らないので問題なし。
・現在のビットより下から余った分を全て持ってくることで、常に最善を尽くす形にできるので、
　それでも足りない分は上から持ってきてもそれで損をすることはない(はず)。
"""

import sys
from collections import Counter

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, M = MAP()
    A = LIST()

    C = Counter(A)    
    ans = 0
    # 下位ビットから見る
    for k in range(0, 61):
        # このビットは必要
        if N & 1<<k:
            if C[1<<k] == 0:
                # なければ上位ビットを見に行く
                for k2 in range(k+1, 31):
                    if C[1<<k2] >= 1:
                        k3 = k2
                        # 立っている場所を見つけたら、必要なところまで分解してくる
                        while k3 > k:
                            C[1<<k3] -= 1
                            C[1<<(k3-1)] += 2
                            k3 -= 1
                            ans += 1
                        break
                else:
                    # 最上位ビットまで見て、持ってこれなければNG
                    print(-1)
                    break
            C[1<<k] -= 1
        # 今見ているビットの値を、1つ上のビットに持っていく
        C[1<<(k+1)] += C[1<<k] // 2
        C[1<<k] = C[1<<k] % 2
    else:
        print(ans)
