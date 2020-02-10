# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1299/submission/70683145
　　　https://codeforces.com/contest/1299/submission/70658911
・自力ならず。
・辞書順最小、スタック
・スタックをうまく使う。
・それぞれの要素について、その値と、何個マージした状態かをセットで持たせる。
・直前のものと比べる、を毎回繰り返しても、既にマージした要素同士は再度見ることがないので
　結局O(N)の範囲内に収まる。
・制約3秒だけどN<=100万が入出力でボトルネックになる。
　いつも通り全部繋げたらダメで、まとまり毎に出力したら通った。
　と思ったら、リストの連結まとめてやったら1回出力でも通った。
　100万回のループだけならそこまで遅くなるとも思えないので、
　100万回のappendとか100万回のprintだとつらい感じなのかも。
　(ちなみに1個ずつ出力もやっぱりダメ。)
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

stack = []
for i, a in enumerate(A):
    # とりあえずスタックに詰める
    stack.append((a, 1))
    # 今回のものが、直前のものより小さければマージする
    while len(stack) >= 2 and stack[-2][0] > stack[-1][0]:
        a1, cnt1 = stack.pop()
        a2, cnt2 = stack.pop()
        merged = (a1*cnt1+a2*cnt2) / (cnt1+cnt2)
        # マージ済のものを、さらに1つ前と比較、を繰り返す
        stack.append((merged, cnt1+cnt2))

# 出力は全部まとめてから'\n'.joinだとダメで、これだと通った
for a, cnt in stack:
    print((str(a) + '\n') * cnt, end='')
# 内ループなくしたら、これでも通った
# ans = []
# for a, cnt in stack:
#     ans += [str(a)] * cnt
# print('\n'.join(ans))
