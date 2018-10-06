# -*- coding: utf-8 -*-

N = int(input())
xN, yN, hN = [0] * N, [0] * N, [0] * N
for i in range(N):
    xN[i], yN[i], hN[i] = map(int, input().split())

# 頂上としてありうる座標101*101を全検索
for y in range(101):
    for x in range(101):
        # 仮の頂上の高さを保持する
        tmpH = 0
        # 実施回数と一致回数の比較用
        cnt1, cnt2 = 0, 0
        # 任意の座標について、N個の情報との矛盾がないかを確認していく
        for i in range(N):
            # 高さ0の座標は高さの比較対象として相応しくないのでいったん無視
            if hN[i] == 0:
                continue
            cnt1 += 1
            # 頂上座標を仮固定しているので、仮の頂上の高さが定まる
            h = hN[i] + abs(x - xN[i]) + abs(y - yN[i])
            # 初回は値を保持
            if tmpH == 0:
                tmpH = h
                cnt2 += 1
            # 2回目以降は保持した初回の値と今回を比較
            else:
                # 一致しなければ矛盾しているので、その座標は終了
                if tmpH != h:
                    break
                else:
                    cnt2 += 1
        # 高さ0の場所について、矛盾がないか確認
        # (tmpHの確定後に全ての場所を試したいので別ループにする)
        for i in range(N):
            if hN[i] == 0:
                cnt1 += 1
                # 仮の高さがマンハッタン距離より大きいのはおかしい
                # (距離のが大きいのは問題ない。本来マイナスなくらい遠くても高さは0スタートだから)
                if tmpH > abs(x - xN[i]) + abs(y - yN[i]):
                    break
                else:
                    cnt2 += 1
        # 実施回数と一致回数が一致していれば正しい情報として出力
        if cnt1 == cnt2:
            print(str(x) + ' ' + str(y) + ' ' + str(tmpH))
            exit()