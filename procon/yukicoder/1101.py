"""
・なんとか自力AC。まじでえげつない、こどふぉBとかで解けなくて激冷えする系のやつ。。
・基本方針は、時間が許す限りすすれるだけすすりながら、残1になったらティッシュ。
　これをひたすらやることを考える。
・まず最初の1回は無条件ですすって0に戻せる。
・この後はティッシュまでの部分を1セットとして考えて、最後に*(P+1)回すればOK、じゃなかった。。
・次にすすれるまでの時間Tは、ティッシュの後でリセットされないので、
　2周目以降、最初にすすれるタイミングが場合によっては早まる。これを考慮しないといけない。
・色々考えた後で気付いたことだけど、リセットされないということは、すすれるタイミングに関しては
　ティッシュと関係なく全時間を1つのものとして考えていいことになる。
・なので、ティッシュまでの周期毎に切り分けて考えて最後にティッシュ回数Pをかけるのではなく、
　最初からティッシュ分を織り込んだV*(P+1)を1つにして考えていく。
・すする周期は(T-1)なのでこれで割るのだが、切り捨てと見せかけて少し違う。しんどい。
　T進んで1戻るみたいな動きを繰り返すので、実質1はみ出る。普通に切り捨てで割ると、
　最後に死んでからすすって1戻るみたいなのをセーフにしてしまうケースが出てくる。(多分)
　これを考慮するために、時間全体から-1して、最終的に(V*(P+1)-1)//(T-1)となる。これがすすれる回数。
　あとは最初の1回と元々ティッシュだけで保てる分の時間を足す。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

V, T, P = MAP()

# 最初の1回すする + 元々いける分 + すすって伸ばせる分
ans = 1 + V*(P+1) + (V*(P+1)-1)//(T-1)
print(ans)
