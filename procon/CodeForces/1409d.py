"""
・なんとか自力AC
・桁和系、繰り上がりの実装
・危なかった。方針は合ってる感じがするのにWAがなかなか取れず。
・結局、繰り上がり処理する時に10になっちゃう桁を正しく処理できてなかった。
・正確には、普通に最後まで処理した時はうまくいくんだけど、
　途中で必要な数まで桁和が下がってbreakした時に10がそのまま残っちゃってた。
・時間も惜しかったので、10になってる桁があるか無理矢理チェックする処理入れて通した。
　なんかもっとキレイに書けそう。。
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
INF = 10**19
MOD = 10**9 + 7

def digit_sum(n):

    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return ans

for _ in range(INT()):
    N, s = MAP()

    dsm = digit_sum(N)
    if dsm <= s:
        print(0)
        continue
    A = list(map(int, str(N)))[::-1]
    M = len(A)
    for i in range(M-1):
        if A[i] == 0:
            continue
        dsm += 1 - A[i]
        A[i] = 0
        A[i+1] += 1
        if dsm <= s:
            break
    for i in range(M-1):
        if A[i] == 10:
            dsm += 1 - A[i]
            A[i] = 0
            A[i+1] += 1
    if A[-1] == 10:
        dsm += 1 - A[-1]
        A[-1] = 0
        A.append(1)
    while dsm > s:
        dsm += 1 - A[-1]
        A[-1] = 0
        A.append(1)
    L = int(''.join(map(str, A[::-1])))
    ans = L - N
    print(ans)
