# -*- coding: utf-8 -*-

import itertools

D, G = map(int, input().split())

# 各点数問題毎の、全問解いた場合の回答数と得点を保持
class ScoreSet:
    def __init__(self, times, score):
        self.times = times
        self.score = score

# 各パターン毎の、合計回答数と合計得点と、どの点数問題を全問解いたのかを保持
class ScoreSetSum:
    def __init__(self, times, score, bool_list):
        self.times = times
        self.score = score
        self.bool_list = bool_list

# 入力内容から、ScoreSetのリストを作成
scoreSetList = []
for i in range(D):
    p, c = map(int, input().split())
    scoreSetList.append(ScoreSet(p, p*100*(i+1) + c))

# どの点数問題を全問解いたのかの各パターンの全網羅
bool_lists = list(itertools.product([True, False], repeat=D))

# パターン全網羅とScoreSetのリストから、ScoreSetSumのリストを作成
scoreSetSumList = []
for bool_list in bool_lists:
    times_tmp = 0
    score_tmp = 0
    for i in range(len(bool_list)):
        if bool_list[i]:
            times_tmp += scoreSetList[i].times
            score_tmp += scoreSetList[i].score
    scoreSetSumList.append(ScoreSetSum(times_tmp, score_tmp, bool_list))

# ScoreSetSumのリストを元に、適切なパターンを探していく
ans = float("inf")
for scoreSetSum in scoreSetSumList:
    tmp = float("inf")
    # 得点が既にG以上であれば、そのままその回答数が候補
    if scoreSetSum.score >= G:
        tmp = scoreSetSum.times
    else:
        # 足りなかったら、全問解いていない点数問題から最も高いもの選ぶ
        for i in range(D-1, -1, -1):
            if scoreSetSum.bool_list[i] == False:
                score = scoreSetSum.score
                times = scoreSetSum.times
                # 選んだ点数問題を1問ずつ解いていく
                for j in range(scoreSetList[i].times):
                    score += 100*(i+1)
                    times += 1
                    # G以上に達した時点で、その時の回答数が候補
                    if score >= G:
                        tmp = times
                        break
                    # 最後まで解いてもG以上にならなければ、このパターンは候補外
                break
    # 今までの最小と今回の候補を比較
    ans = min(ans, tmp)

print(ans)