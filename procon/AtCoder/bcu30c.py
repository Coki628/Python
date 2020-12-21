# -*- coding: utf-8 -*-

"""
・自力AC！
・愚直に全部見ようとすると4文字*300種類あるので300^4で無理。
・そこで、右上と左下を先に固定すると、内側では左上との関係と右下との関係を独立に確認して
　最後に両方のうまくいった数を掛ければよくなるので300^3=2700万にできる。
・単語の存在確認をsetのtupleキーで確認したらTLEで、2次元配列にしたらAC。
・ループはi,jの前後逆をまとめて計算量約半分にしたらこれも効果高い。pypyAC1.5秒→0.9秒。
・2次元配列を1次元+ビットにしてみたらこれは却って遅くなってTLE。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
table = list2d(N, N, 0)
for _ in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    table[a][b] = 1

ans = 0
# 右上と左下をi,jとして先に固定
for i in range(N):
    for j in range(i, N):
        # 左上との関係
        cnt1 = 0
        for k in range(N):
            # 両方の単語が存在すれば、この組み合わせは可能
            if table[k][i] and table[k][j]:
                cnt1 += 1
        # 右下との関係
        cnt2 = 0
        for k in range(N):
            if table[i][k] and table[j][k]:
                cnt2 += 1
        if i == j:
            # 左上,左下,右上でうまくいく数 * 左下,右上,右下でうまくいく数
            ans += cnt1 * cnt2
        else:
            ans += cnt1 * cnt2 * 2
print(ans)
