"""
・終了10秒前提出ACでマジのブザービート。Unratedなんだけどね。
・再帰で全探索、ビットでの集合管理
・最初、問題読み違えて貪欲みたいのしようとしてダメで、
　制約の小ささと、使えなくなった要素を管理する必要があるのとでbitDPに向かったけどこれが。。
・YES/NOや通り数とかでなく、答えとなる文字列を1個構築する必要があったので、復元まで必要みたいな
　修羅の道に走りかけて思い直した。
・同じく制約の小ささと、答えを1つ示すって所から再帰で探して帰りがけに構築が妥当かもと判断。
　多分15分くらい？で組んで提出、無事AC。
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

def rec(i, used):
    if i == M:
        return True
    for c in se[i]:
        # 欲しい集合は、位置iで文字cを取った時に重複「しない」文字列なので、マスクとxorする
        if used & (mask^prep[i][c]) == 0:
            if rec(i+1, used|(mask^prep[i][c])):
                ans.append(c)
                return True
    return False

for _ in range(INT()):
    N, M = MAP()
    A = [[ord(c)-97 for c in input()] for i in range(N)]

    prep = list2d(M, 26, 0)
    for i in range(M):
        for j in range(N):
            prep[i][A[j][i]] |= 1<<j

    se = [set() for i in range(M)]
    for S in A:
        for i, s in enumerate(S):
            se[i].add(s)

    ans = []
    mask = (1<<N) - 1
    if rec(0, 0):
        ans = [chr(c+97) for c in ans[::-1]]
        print(''.join(ans))
    else:
        print(-1)
