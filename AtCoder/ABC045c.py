# -*- coding: utf-8 -*-

S = list(input())
N = len(S)-1

ans = 0
# 間に+を入れるかどうかでビット全検索
for i in range(1 << N):
    offset = 0
    for j in range(N):
        if (i>>j) & 1 == 1:
            # 該当のビットが1なら+を挿入
            S.insert(j+1+offset, '+')
            # +を入れた分リストが長くなるので、次回の位置を調整
            offset += 1
            # リスト→文字列→式にして答えに足す
    ans += eval(''.join(S))
    # +を削除
    S = [j for j in S if j != '+']

print(ans)