# -*- coding: utf-8 -*-

"""
stackの応用(水たまりの面積)
"""

from collections import deque

S = list(input())

# 総合面積
ans = 0
# 総合面積用スタック(直近の\の位置)
stack1 = deque()
# 個別用スタック(その水たまりを開始する\の位置, その時点での面積)
stack2 = deque()
for i in range(len(S)):
    if S[i] == '\\':
        stack1.append(i)
    # 一番最近いれた\を取り出せば、今回の/と対になる(同じ高さ)
    elif S[i] == '/' and stack1:
        # 両者の距離がその高さでの面積になる
        # (\/が三角形な分-1なのが半開区間だからちょうど合う)
        j = stack1.pop()
        ans += i - j

        # 各水たまりのマージ作業
        # stack2から、今回の合併候補を抽出
        # (条件：水たまりの開始位置が今回対になってる\より右側にある)
        pond = i - j
        while stack2 and stack2[-1][0] >= j:
            pond += stack2.pop()[1]
        # マージした水たまりの情報を詰める
        stack2.append([j, pond])
print(ans)
# リスト内包とアンパックできれいにまとまった
print(len(stack2), *[pond for j, pond in stack2])
