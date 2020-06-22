from collections import defaultdict, Counter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self):
        # 先验概率，形式，以西瓜书中判断好瓜坏瓜为例，{"是":0.471,"否":0.529}
        self.prior_prob = defaultdict(float)

        # 条件概率，形式：{"是":{"色泽":{"青绿":0.375}},"否":{"色泽":{"青绿":0.333}}}
        self.condi_prob = defaultdict(defaultdict)

        # 存储每个类别样本数
        self.n_label = defaultdict(float)

        # 特征可能的取值数
        self.S_j = defaultdict(float)

        # 每个类别中的样本
        self.label_idx = defaultdict(list)

    def fit(self, X, y):
        n_samples, n_features = X.shape[0], X.shape[1]
        self.n_label = dict(Counter(y))

        # 计算先验概率，防止出现概率为0的情况，做一个拉普拉斯平滑
        for label, num in self.n_label.items():
            self.prior_prob[label] = (num+1) / (n_samples+len(self.n_label))

        # 统计每个类别中的样本的索引
        for i in range(n_samples):
            self.label_idx[y[i]].append(i)

        # 计算条件概率
        for label, idx in self.label_idx.items():
            data_label = X[idx]
            label_condi_prob = defaultdict(defaultdict)
            for i in range(n_features):
                # 获取第i列特征所有可能的取值以及取值对应的数量
                data = data_label[:, i]
                feature_prob = defaultdict(float)
                feature_count = dict(Counter(data))
                self.S_j[i] = len(feature_count)
                for feature, num in feature_count.items():
                    feature_prob[feature] = (num + 1) / (self.n_label[label] + self.S_j[i])
                label_condi_prob[i] = feature_prob
            self.condi_prob[label] = label_condi_prob

    def predict(self, x):
        # 计算后验概率
        post_prob = defaultdict(float)
        for label, condi_prob in self.condi_prob.items():
            prob = np.log(self.prior_prob[label])
            for i, feature_val in enumerate(x):
                if feature_val in condi_prob[i]:
                    prob += np.log(condi_prob[i][feature_val])
                else:
                    prob += np.log(1/(self.n_label[label] + self.S_j[i]))
            post_prob[label] = prob
        return max(post_prob, key=lambda x: post_prob[x])
        # return post_prob

    def score(self, X_test, y_test):
        right = 0
        for x, y in zip(X_test, y_test):
            label = self.predict(x)
            if label == y:
                right += 1
        return right / float(len(X_test))


if __name__ == '__main__':
    # datasets = np.array([['青年', '否', '否', '一般', '否'],
    #                      ['青年', '否', '否', '好', '否'],
    #                      ['青年', '是', '否', '好', '是'],
    #                      ['青年', '是', '是', '一般', '是'],
    #                      ['青年', '否', '否', '一般', '否'],
    #                      ['中年', '否', '否', '一般', '否'],
    #                      ['中年', '否', '否', '好', '否'],
    #                      ['中年', '是', '是', '好', '是'],
    #                      ['中年', '否', '是', '非常好', '是'],
    #                      ['中年', '否', '是', '非常好', '是'],
    #                      ['老年', '否', '是', '非常好', '是'],
    #                      ['老年', '否', '是', '好', '是'],
    #                      ['老年', '是', '否', '好', '是'],
    #                      ['老年', '是', '否', '非常好', '是'],
    #                      ['老年', '否', '否', '一般', '否'],
    #                      ])
    # X = datasets[:, :-1]
    # y = datasets[:, -1]
    # clf = NaiveBayes()
    # clf.fit(X, y)
    # correct_num = 0
    # print(dict(clf.condi_prob))
    # for i in range(len(X)):
    #     predict = clf.predict(X[i])
    #     print(predict)
    #     if y[i] == predict:
    #         correct_num += 1
    # correct_rate = round(correct_num / len(y) * 100, 2)
    # print("正确率：%f" % correct_rate)
    data = load_iris()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=1)
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

