# -*- coding: UTF-8 -*-
import jieba              #分词函数
file_train = open("user_tag_query.2W.TRAIN.csv", encoding="utf-8") #训练集文本
#stopwords = [x.strip() for x in open("stopword.txt", encoding="utf-8").readlines()]
contents=file_train.readlines()
count=0
for line in contents:
    count = count+1
    flag_age       =  line.split()[1]
    flag_gender    =  line.split()[2]
    flag_education =  line.split()[3]              #获得类别标签作为文件名
    lines          =  line.split()[4:]
    content=" ".join(lines)
    word=jieba.cut(content, cut_all=True)
    string=" ".join(word)
    print(count)
    #string = " ".join(list(set(word) - set(stopwords)))
    # 15000训练文件地址变量 把一类文件放到一个文档中


    train_age = "train_data/train_age/" + flag_age + "/" + str(count) + ".txt"
    train_gender = "train_data/train_gender/" + "/" + flag_gender + "/" + str(count) + ".txt"
    train_education = "train_data/train_education/" + flag_education + "/" + str(count) + ".txt"

    names = [train_age, train_education, train_gender]
    for name in names:
        file_fenci = open(name, "a", encoding="utf-8")  # 分词文件
        file_fenci.write(string)
        file_fenci.close()
file_train.close()