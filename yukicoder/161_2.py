"""
参考：https://www.slideshare.net/kmjp/yukicoder-no161
・想定解
・貪欲
・順番関係なく、勝てるだけ勝ってあいこにできるだけする。
・これだと制約10万でも通るもんね。本来こっちが出てくるべきなんだよねー。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

g, c, p = MAP()
S = input()
N = len(S)

# 順番は関係ないので回数だけ集計
C1 = { 'G': g, 'C': c, 'P': p }
C2 = Counter(S)
ans = 0
# 勝てるだけ勝つ
for win, lose in zip(['G', 'C', 'P'], ['C', 'P', 'G']):
    if C1[win] >= C2[lose]:
        C1[win] -= C2[lose]
        ans += C2[lose] * 3
        C2[lose] = 0
    else:
        C2[lose] -= C1[win]
        ans += C1[win] * 3
        C1[win] = 0

# あいこも同様
for same in ['G', 'C', 'P']:
    if C1[same] >= C2[same]:
        C1[same] -= C2[same]
        ans += C2[same]
        C2[same] = 0
    else:
        C2[same] -= C1[same]
        ans += C1[same]
        C1[same] = 0
print(ans)
