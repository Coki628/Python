# -*- coding: utf-8 -*-

# 再帰関数でやってみる版(多分dfsなのかと)

N = int(input())

def f(rest, s):
    # 最後の文字まで行ったら出力
    if rest == 0:
        print(s)
    else:
        for c in ['a','b','c']:
            # sに1文字追加して再帰呼び出し
            # (aから順番に深く潜って、所定の文字数になった所で出力してくれる)
            f(rest-1, s+c)
# 最初の呼び出し(最初は空文字で、''+'a'から始まる)
f(N, '')