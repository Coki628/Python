"""
・終了後AC。参戦中、WAしてちょっと悩んだから飛ばした。
・結局の所、パリティのペアでほぼ決まってしまって、あぶれるとしたら偶奇それぞれ1つずつだけ。
　なのでそうなった時だけ距離1のペアが1つでも作れるかを判定すればいい。
・先に距離1ペアを固めてからパリティ見ようって方針に走ったのが失敗だった。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()
    
    cnte = cnto = 0
    for i in range(N):
        if A[i] % 2 == 0:
            cnte += 1
        else:
            cnto += 1
    if cnte % 2 == 0 and cnto % 2 == 0:
        YES()
    else:
        A.sort()
        i = 0
        while i < N-1:
            if A[i+1] - A[i] == 1:
                YES()
                break
            else:
                i += 1
        else:
            NO()
