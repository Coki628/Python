"""
・終了後自力AC
・中央値、2進数
・全部の値を陽に持ったらもちろん無理なので、最初の中央値を持っておいて、
　削除する値の大小に応じて中央値をずらしていく方針を取る。
・ある値を削除した時、中央値が変化するかどうかは、削除する値の大小と、残っている値の数の偶奇による。
・中央値を変化させる時、動かす先の値がまだ残っているかどうかを管理するのにsetを使う。
・これはやれば時間内解けたなーって感じだけど、多分時間あってもGで詰まってここに届いてなさそう。。
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
    N, M = MAP()
    # 大小比較のやりやすさのため、10進数変換しておく
    A = [int(input(), 2) for i in range(N)]

    mid = 2**M // 2 - 1
    used = set()
    ln = 2**M
    for a in A:
        # 中央値を大きい方に動かす
        if a <= mid and ln % 2 == 0:
            tmp = mid + 1
            while tmp in used:
                tmp += 1
            mid = tmp
        # 中央値を小さい方に動かす
        if a >= mid and ln % 2 == 1:
            tmp = mid - 1
            while tmp in used:
                tmp -= 1
            mid = tmp
        # 削除済と残っている長さを更新
        used.add(a)
        ln -= 1
    # 2進数に戻してM桁0埋め
    ans = format(mid, 'b').zfill(M)
    print(ans)
