"""
参考：https://twitter.com/ryusei_ishika/status/1266046817948413952
　　　https://twitter.com/TT_beginner/status/1266047099474292737
・自力ならず。
・二分探索、誤差
・偶数回目はずっと同じ値。奇数回目は単調減少、ずっと下がり続けて、偶数回目の値に近づく。(下回らない)
　これは実験すればすぐ分かったはず。やればよかった。。
・ただしこれが分かっても誤差がつらい。EPS使ってもダメで、
　Decimalで(遅いので)探索回数減らして何とかpythonAC1.2秒。pypyはDecimal遅すぎるのでTLE。
・計算量は3万テストケースにlogで20回くらいだから約60万。
"""

import sys
from decimal import Decimal

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
INF = 10 ** 10
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

# m*2+1回カップを入れた時、温度がt以上かどうか
def check(m):
    res = (h*(m+1) + c*m) / (m*2+1)
    return res >= t

for _ in range(INT()):
    h, c, t = map(Decimal, input().split())

    res = bisearch_max(0, 10**6, check)
    if res == 10**6-1:
        print(2)
    else:
        mn = INF
        ans = -1
        # Decimalは正確だからか、前後2値の確認だけでもACした。
        for m in range(res, res+2):
            res = (h*(m+1) + c*m) / (m*2+1)
            diff = abs(res - t)
            if diff < mn:
                mn = diff
                ans = m*2 + 1
        print(ans)
