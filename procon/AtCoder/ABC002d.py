# coding=utf-8
 
from itertools import combinations as comb
 
# n:議員数、m:人間関係の数
n, m = map(int, input().split())
 
# 人間関係を格納するメソッド
def createRelations():
    # 中身が全部Falseの隣接行列の作成
    created_relation = [[False] * n for _ in range(n)]
    for i in range(m):
        rel1, rel2 = map(int, input().split())
        rel1 -= 1
        rel2 -= 1
        # 人間関係がある所はTrueにする
        created_relation[rel1][rel2] = True
        created_relation[rel2][rel1] = True
    return created_relation
 
# お互いに人間関係があるかを確認するメソッド
def confirmRelations():
    answer = 1
    # 最初に条件に合致するものが最大であるようにしたいので、大きい方から見ていく
    for relation_size in range(n, 0, -1):
        # 議員数に応じた関係サイズの組み合わせを作る
        all_relations = comb(range(n), relation_size)
        # 関係サイズの組み合わせを使って、各人の人間関係を調べていく
        for relation in all_relations:
            # createRelationsで作った隣接行列から、組み合わせに当てはまる所だけ集めたリストを作る
            judge = [relations[i][j] for (i, j) in comb(relation, 2)]
            # リスト内が全てTureなら関係サイズをanswerに入れる
            if all(judge):
                answer = relation_size
        # 候補が見つかったら、もう探す必要はないのでbreak
        if answer != 1:
            break
    return answer

relations = createRelations() 

print(confirmRelations())