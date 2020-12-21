# -*- coding: utf-8 -*-

"""
・終了後20分くらいで自力AC。。悔しい。。
・(連続しない)部分文字列、累積で前計算。
・各文字毎に、次回出現位置を覚えておいてO(1)で飛べるようにしておく。
・Bで悩みすぎたな。もう少し早くこっちと向き合っておけば良かった。
・ツイート情報だけど、確かにABC138eとかなり類題感あるな。
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
    
    S = [ord(c) - 97 for c in input()]
    T = [ord(c) - 97 for c in input()]
    N = len(S)
    M = len(T)

    C1 = Counter(S)
    C2 = Counter(T)
    for k, v in C2.items():
        # 例外処理
        if C1[k] == 0:
            print(-1)
            break
    else:
        # アルファベット26文字についての次回出現位置
        acc = list2d(26, N+1, INF)
        for i, c in enumerate(S):
            # 1つ手前に出現位置を打ち込む
            acc[c][i] = i + 1
        for c in range(26):
            for i in range(N-1, -1, -1):
                # 後ろから累積min
                acc[c][i] = min(acc[c][i], acc[c][i+1])
        cnt = 1
        i = 0
        for c in T:
            # この先にもうない文字だったら最初に戻る
            if acc[c][i] == INF:
                cnt += 1
                i = 0
            # 該当文字の場所まで飛ぶ
            i = acc[c][i]
        print(cnt)
