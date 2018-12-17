# -*- coding: utf-8 -*-

"""
参考：https://misteer.hatenablog.com/entry/ABC113
・DPとビット全検索
"""

H, W, K = map(int, input().split())
# 0-indexed
K -= 1
MOD = 10 ** 9 + 7

dp = [[0] * W for i in range(H+1)]
# 縦棒の長さ0(つまり横棒なし)、一番左の棒を下りてくる1つ
dp[0][0] = 1

for h in range(1, H+1):
    # 縦棒間の数でビット全検索
    for b in range(1 << (W-1)):
        # 横棒が隣接していないチェック
        s = format(b, 'b')
        if s.count('11'):
            continue
        # ある始点の今回の移動先を保持する
        perm = [0] * W
        for i in range(W):
            perm[i] = i
        # 各縦棒間に横棒があるかループ
        for i in range(W-1):
            # あるビットが立っているかどうかで判別
            # (あみだが左始まりだから、ビットも左から始まってると思った方がイメージしやすいかも。
            # それかあみだが右から始まってると思うか。)
            if b >> i & 1:
                # 横棒があればひとつ先とスワップ
                perm[i], perm[i+1] = perm[i+1], perm[i]
        for i in range(W):
            # 前回のW-1ビット分を今回のW-1ビット分の回数足し合わせてくから、最終的に膨大になる
            dp[h][perm[i]] += dp[h-1][i]
            dp[h][perm[i]] %= MOD

print(dp[H][K])
