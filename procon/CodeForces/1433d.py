"""
・なんとか自力AC。結構悩んだ。
・木、グラフ
・最初は1直線グラフに貪欲につないでくのとか考えたんだけどうまくいかなくて、
　図書いてどうするかぐりぐり考えてたら、1人が自分と違うグループのやつ全員と繋がれば、
　自分と同じグループのやつ以外は全てOKになると気付く。同じように、
　違うグループのやつを1人用意して、そいつはさっきの最初のやつと同じグループのやつ全員と繋がれば、
　これで全員連結になるんじゃないかと思う。こうなると、全員同じグループの場合以外は全部可能になる。
・この考察は合ってて、無事AC。でもこの問題、制約がN^2を許すので、
　全ペアについてつなげるかチェックしながらUnionFindで連結判定して繋がってないとこ全部繋ぐ、
　でもよかったみたい。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()

    grp1 = A[0]
    v1 = 0
    v2 = -1
    for i in range(N):
        if A[i] != grp1:
            v2 = i
            break
    if v2 == -1:
        NO()
        continue

    ans = []
    for i in range(1, N):
        if A[i] != grp1:
            ans.append((v1+1, i+1))
        else:
            ans.append((v2+1, i+1))
    YES()
    for a in ans:
        print(*a)
