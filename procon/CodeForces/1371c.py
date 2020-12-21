"""
参考：ツイート
・自力ならず。。
・貪欲シミュレーション
・3問連続O(1)考察で精神的に疲れてしまったからか、コンテスト中は何も考えつかなかった。
・言われてみれば、逆張りを先に消化しないとうまくいかない、の一言に尽きる。。
・制約(N<=10^18でさらに複数テストケース)的に、どっちかに先に全部渡すくらいしか有効っぽい手段がない、
　くらいのことは考えるべきだったな。。Cだし何か難しく数学しなきゃ、な気がして心を閉ざした感はある。。
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
    a, b, N, M = MAP()

    if a > b:
        a, b = b, a
    # とりあえず逆張り勢全員に渡す
    a -= M
    # 足りなくなったらNG
    if a < 0:
        No()
        continue
    # 残りを順張り勢に渡して、足りるか確認
    if N <= a + b:
        Yes()
    else:
        No()
