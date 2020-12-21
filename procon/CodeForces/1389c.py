"""
・自力AC
・貪欲、全探索
・色々ガチャガチャしてると、成功するのは(多分)0000みたいのか1010みたいのしかないと分かる。
・ある数字をa,bとして、0〜9のa,bを2重ループで全部試す。
・N<=20万で、10^2=100のループ回しても2000万、間に合う。
"""

import sys
from string import digits

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
    S = input()
    N = len(S)

    ans = INF
    for a in digits:
        for b in digits:
            is_a = True
            cnt = 0
            for i in range(N):
                if is_a and S[i] == a:
                    is_a = False
                    cnt += 1
                elif not is_a and S[i] == b:
                    is_a = True
                    cnt += 1
            # 違う数字だったら、偶数長しか使えない
            if a != b:
                cnt -= cnt & 1
            ans = min(ans, N-cnt)
    print(ans)
