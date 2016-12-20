
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import jieba
import numpy as np
import time
#训练集分词

def predict_test(text):
    line          =  text.split()[1:]
    content=" ".join(line)
    word=jieba.cut(content)
    string=" ".join(word)
    temporary_file=open("data/0/1.txt","w",encoding="utf-8")
    temporary_file.write(string)
    temporary_file.close()
#tftf_idf计算  构造分类器
def naive_bayes(data_train,count_vect,transformer):
    # 词频计算 加载分词信息 计算tf—idf
    rawData_train=count_vect.fit_transform(data_train.data)#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tf_idf = transformer.fit_transform(rawData_train)
    #naive_bayes训练分类器
    modle = MultinomialNB(0.01).fit(tf_idf, data_train.target)
    return modle
#预测
def attribute_predict(modle,data_test,count_vect,transformer):
   rawData_test = count_vect.transform(data_test.data)
   tf_idf = transformer.transform(rawData_test)
   #print(tf_idf)
   result_predict=modle.predict(tf_idf)
   return result_predict
#main
if __name__ == '__main__':
    start=time.clock()
    count_vect = CountVectorizer()  #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 类调   该类会统计每个词语的tf-idf权值
    print("****************************************")
    age_data_train = datasets.load_files("train_data/train_age", encoding="utf-8")
    modle_age = naive_bayes(age_data_train, count_vect, transformer)
    file_test = open("user_tag_query.2W.TRAIN.csv", encoding="utf-8")
    contents = file_test.readlines()
    result_age=[]
    for line in contents:
       predict_test(line)
       age_data_test = datasets.load_files("data", encoding="utf-8")
       predict_age = attribute_predict(modle_age, age_data_test, count_vect, transformer)
       result_age.append(predict_age)
    #for i in range(0,10):
    #  print(result_age[i])
    end_age=time.clock()
    print(end_age-start)
    print("****************************************")

    gender_data_train = datasets.load_files("train_data/train_gender", encoding="utf-8")
    modle_gender = naive_bayes(gender_data_train, count_vect, transformer)
    file_test = open("user_tag_query.2W.TRAIN.csv", encoding="utf-8")
    contents = file_test.readlines()
    result_gender=[]
    for line in contents:
        predict_test(line)
        gender_data_test = datasets.load_files("data", encoding="utf-8")
        predict_gender = attribute_predict(modle_gender, gender_data_test, count_vect, transformer)
        result_gender.append(predict_gender)
    #print(result_gender)
    end_gender=time.clock()
    print(end_gender-start)
    print("****************************************")
    education_data_train = datasets.load_files("train_data/train_education", encoding="utf-8")
    modle_education = naive_bayes(education_data_train, count_vect, transformer)
    file_test = open("user_tag_query.2W.TRAIN.csv", encoding="utf-8")
    contents = file_test.readlines()
    result_education=[]
    for line in contents:
        predict_test(line)
        education_data_test = datasets.load_files("data", encoding="utf-8")
        predict_education=attribute_predict(modle_education, education_data_test, count_vect, transformer)
        result_education.append(predict_education)
    end_education=time.clock()
    print(end_education-start)
    print("******************************")
    flag_id = []
    file_test = open("user_tag_query.2W.TRAIN.csv", encoding="utf-8")
    file_test_txt = open("test.txt", "a", encoding="utf-8")
    contents = file_test.readlines()
    for line in contents:
        flag_id.append(line.split()[0])  # 获得id作为文件名
    age = np.array(result_age)
    gender = np.array(result_gender)
    education = np.array(result_education)
    id=np.array(flag_id)
    for i in range(0,len(flag_id)):
       file_test_txt.write(str(flag_id[i]).strip("[ ]")+" "+
                           str(result_age[i]).strip("[ ]")+" "+
                           str(result_gender[i]).strip("[ ]")+" "+
                           str(result_education[i]).strip("[ ]")+"\n")
       print(str(flag_id[i])+str(result_age[i])+str(result_gender[i])+
             str(result_education[i]))
    end=time.clock()
    print(end-start)



