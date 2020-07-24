"""
・青diff600点自力AC！嬉しい。
・(連続でない)部分文字列
・一見、全体に対する前計算が必要そうな量にも見えるが、各クエリにO(N)かけていい。
　以下、各クエリ内での処理。
・先に後ろから「あるDからx文字以内にあるMの数」を前計算しておくと、
　前から見て行った時に、あるDが使えなくなった時にDMペアが何個使えなくなるか、
　がすぐに分かるようになる。
・あとはD,M,Cの出現に従って、Dの数,DMの数,DMCの数を加減させていけばいい。
・多分、logを落とすためなんだろうけど制約がきつい。(100万のN*75クエリ)
　pypyTLEでC++で書き直してAC0.96秒。
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

N = INT()
S = input()

def f(x):
    # DM[i] := 位置iにあるDからx文字以内にあるMの数
    DM = [0] * N
    mcnt = 0
    for i in range(N-1, -1, -1):
        if i+x < N:
            if S[i+x] == 'M':
                mcnt -= 1
        if S[i] == 'M':
            mcnt += 1
        elif S[i] == 'D':
            DM[i] = mcnt

    dcnt = dmcnt = dmccnt = 0
    for i in range(N):
        if i-x >= 0:
            if S[i-x] == 'D':
                dcnt -= 1
                dmcnt -= DM[i-x]
        if S[i] == 'D':
            dcnt += 1
        elif S[i] == 'M':
            dmcnt += dcnt
        elif S[i] == 'C':
            dmccnt += dmcnt
    return dmccnt

INT()
for q in LIST():
    print(f(q))
