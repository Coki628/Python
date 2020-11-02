"""
・自力ならず。。これは解きたかった。。。
・シミュレーション
・題意を満たすには、単調増加が続く限り同じ親に付け続けるのが最善。
　あとはこの操作をどうプログラムに落とし込むか。コンテスト中はうまくまとめきれず。。
・翌朝気持ち切り替えてやってみたら結構ちゃんとできた。
　頂点を管理するリストを3次元にしたらきれいにまとまった。
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

    A.reverse()
    # nodes[i][j][k] := 深さiで、共通の親を持つグループjの頂点k
    nodes = [[[A.pop()]]]
    ans = 0
    while A:
        curdepth = []
        # 1つ前の深さにある頂点を全部回す
        for pars in nodes[-1]:
            for par in pars:
                # 最初の1つは必ず取れるので番兵入れておく
                chlds = [-1]
                while A and A[-1] > chlds[-1]:
                    chlds.append(A.pop())
                curdepth.append(chlds[1:])
            nodes.append(curdepth)
        ans += 1
    print(ans)
