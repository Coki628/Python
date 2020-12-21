"""
参考：https://img.atcoder.jp/intro-heuristics/editorial.pdf
・公式解の高速化
・データの持たせ方を工夫して、差分の計算は等差数列の和、状態の更新は隣接リストへの追加と削除で行う。
・計算量的には状態の更新に隣接リストの長さ分かかるけど、全体の要素数としてD個しかないのは変わらないので、
　実際は最大の長さよりは全然小さいものとみなせる。(ここがボトルネックになるような大きさならC++のset使うかな。)
・これだとだいぶ速くて、pythonでもAC0.49秒。
"""

import sys
from bisect import bisect_left

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

D = INT()
M = 26
C = LIST()
S = [[]] * (D+1)
S[0] = [0] * M
for i in range(1, D+1):
    S[i]  = LIST()
adjli = [[0] for i in range(M)]
T = [0] + [t-1 for t in LIST(D)]
for d, t in enumerate(T[1:], 1):
    adjli[t].append(d)
for i in range(M):
    adjli[i].append(D+1)

def check(T):
    score = 0
    for a in range(M):
        for i in range(1, len(adjli[a])):
            curd = adjli[a][i-1]
            nxtd = adjli[a][i]
            cnt = nxtd - curd
            score += S[curd][a]
            score -= C[a] * get_sum(0, 1, cnt)
    return score

# day日目のコンテストをaからbに変更する
def change(day, a, b):
    res = 0
    # コンテストaのコストを再計算
    res -= S[day][a]
    di = adjli[a].index(day)
    for i in range(di, di+2):
        curd = adjli[a][i-1]
        nxtd = adjli[a][i]
        cnt = nxtd - curd
        res += C[a] * get_sum(0, 1, cnt)
    adjli[a].pop(di)
    curd = adjli[a][di-1]
    nxtd = adjli[a][di]
    cnt = nxtd - curd
    res -= C[a] * get_sum(0, 1, cnt)

    # コンテストbのコストを再計算
    res += S[day][b]
    di = bisect_left(adjli[b], day)
    curd = adjli[b][di-1]
    nxtd = adjli[b][di]
    cnt = nxtd - curd
    res += C[b] * get_sum(0, 1, cnt)
    adjli[b].insert(di, day)
    for i in range(di, di+2):
        curd = adjli[b][i-1]
        nxtd = adjli[b][i]
        cnt = nxtd - curd
        res -= C[b] * get_sum(0, 1, cnt)

    return res

score = check(T)
Q = INT()
for i in range(Q):
    d, q = MAP()
    q -= 1

    score += change(d, T[d], q)
    print(score)
    T[d] = q
