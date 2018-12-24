# -*- coding: utf-8 -*-

"""
参考：http://garnacha.techblog.jp/archives/39562260.html
　　　http://d.hatena.ne.jp/simezi_tan/20130903/1378179426
　　　https://babcs2035.hateblo.jp/entry/2018/12/02/144726
・確率DP、各対戦相手＆自分の前回対戦までの勝率を貰うDP
・ビット演算でマスクする
・pythonでTLEしてpypyでAC
・計算量K*(2^K)^2 = 10*(2^10)^2だと10*1000*1000で1000万くらいだからまあ大変か。
"""

def battle(a, b):
    return 1 / (1 + pow(10, (b-a)/400))

K = int(input())
R = [int(input()) for i in range(1 << K)]

# 人iがj回戦目で勝つ確率
dp = [[1] * (K+1) for i in range(1 << K)]

# 各人より各対戦を先に終わらせていきたいのでiとjのループが逆
for j in range(1, K+1):
    for i in range(1 << K):
        sm = 0
        for k in range(1 << K):
            # 下位j-1ビット目の不一致確認用(確認したいビットだけ立てる)
            mask1 = 1 << j-1
            # 下位jビット目より上位の一致確認用(確認したい範囲のビットだけ立てる)
            mask2 = 2**K-1 << j
            # 条件に合えば、対戦相手となりうる
            # (両方に同じマスクかければ、任意のビットの一致も全ビットの一致も確認できる)
            if (i & mask1 != k & mask1
                    and i & mask2 == k & mask2):
                # その対戦相手がここまで勝ち上がってくる確率
                # * 今回の対戦での勝率 を足し合わせていく
                sm += dp[k][j-1] * battle(R[i], R[k])
        # 最後に自分がここまで来る確率と掛け合わせる
        dp[i][j] = dp[i][j-1] * sm

for i in range(1 << K):
    # 小数以下小さいとeとか出るからフォーマット整える
    print('{:.9f}'.format(dp[i][K]))
