"""
参考：https://yukicoder.me/problems/no/374/editorial
　　　http://mayokoex.hatenablog.com/entry/2016/06/05/100800
・自力ならず。全然分かんなかった。。
・ゲーム
・対称な状況を作り相手の真似をする戦略、というのがある。
　今回はそれを適用することで、初手さえ置ければ先手必勝の状況を作れる。
・具体的には、初手を中心に置いて、後は全部後手番の対象位置に置けばいい。
　…聞けばああなるほど、だけど、気づくかな、こんなの。。
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

ar, br = MAP()

if ar >= br:
    print('S')
else:
    print('K')
