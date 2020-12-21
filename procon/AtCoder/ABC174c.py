"""
参考：https://twitter.com/kyopro_friends/status/1289919703637299201
・自力ならず。。ひっさびさにC落とした。。
・倍数判定
・7,77,777,7777とどこまでもでかいのが考えられるので、まともに取り合っては解けない。
・ようはKの倍数かどうかの判定ができればいいので、mod Kで値を持っていれば値が大きくならずに済む。
・あとは桁DPっぽいのでやるみたいに、*10と+7で1桁増やすのを最大K回まで繰り返してチェックすればOK。
　(鳩の巣的な話で、Kまでの値をK回通れば絶対どっかで被るのでそこで周期になる。)
・このCむずくない？てか公式解説の説明がむずい。。フレンズさんのツイートで、mod Kで進めて倍数判定、
　がしっくりきたから何とかそれで解いたけど、それにしてもこの考察でCだし3000人解くんだなぁ。。
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

K = INT()

cur = 7
for i in range(1, K+1):
    cur %= K
    if cur == 0:
        print(i)
        break
    cur *= 10
    cur += 7
else:
    print(-1)
