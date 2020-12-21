"""
・自力AC！
・状態をうまくまとめてDP
・基本的にある値に0〜P-1までの値を足すか掛けるかした時、実はちょうど0〜P-1のどれかに均等に遷移する。
・遷移元が0の時だけは、掛け算は全部0、足し算は均等に遷移する。
・このことから、各操作時の状態としては「0」か「0以外」かだけ持てばよいと分かる。
・あとは上記の性質に従った遷移をさせるDPを書けばOK。
・計算量はKが100万だけど、pythonACギリギリ1.97秒、pypyAC0.08秒。
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

P, K = MAP()

# dp0[i] := i回操作して、今の値が0の通り数
dp0 = [0] * (K+1)
# dp1[i] := i回操作して、今の値が0以外の通り数
dp1 = [0] * (K+1)
dp0[0] = 1
for i in range(K):
    # 0から0への遷移、掛け算P回と足し算1回
    dp0[i+1] += dp0[i] * (P+1)
    # 0から他への遷移、足し算P-1回
    dp1[i+1] += dp0[i] * (P-1)
    # 他から0への遷移、足し算・掛け算1回ずつ
    dp0[i+1] += dp1[i] * 2
    # 他から他への遷移、0に行った2回以外全部
    dp1[i+1] += dp1[i] * (2*P-2)
    dp0[i+1] %= MOD
    dp1[i+1] %= MOD
# 最終的に0なのが答え
ans = dp0[K]
print(ans)
