import pandas as pd
import numpy as np
import re
import os
import logging
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB  # 导入多项式朴素贝叶斯
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置 matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def preprocess_data(data):
    """
    数据预处理函数
    :param data: 原始数据
    :return: 处理后的训练集和测试集
    """
    # 划分数据集
    input_columns = ['发酵容器', '制曲原料', '原料', '生产工艺',
                     '发酵周期', '发酵温度', '用曲量', '制曲温度', '发酵次数', '酒精度', '总酸', '总酯', '固性物']
    X = data[input_columns]
    y_flavor = data['风味']

    X_train, X_test, y_train_flavor, y_test_flavor = train_test_split(X, y_flavor, test_size=0.2, random_state=42)

    # 处理目标变量（在划分后进行标签化）
    unique_flavors = y_train_flavor.unique()
    flavor_labels = {flavor: idx for idx, flavor in enumerate(unique_flavors)}

    y_train_flavor_encoded = y_train_flavor.map(flavor_labels)
    y_test_flavor_encoded = y_test_flavor.map(lambda x: flavor_labels.get(x, -1))

    # 处理输入特征
    def parse_range(value):
        if isinstance(value, str):
            numbers = re.findall(r'\d+\.?\d*', value)
            if len(numbers) == 1:
                return float(numbers[0])
            elif len(numbers) == 2:
                start = float(numbers[0])
                end = float(numbers[1])
                return np.random.uniform(start, end)
            else:
                return np.nan
        elif pd.isna(value):
            return np.nan
        else:
            return float(value)

    def parse_numeric_with_unit(value):
        if isinstance(value, str):
            if '≥' in value:
                number = re.findall(r'\d+\.?\d*', value)
                return float(number[0]) if number else np.nan
            else:
                return parse_range(value)
        elif pd.isna(value):
            return np.nan
        else:
            return float(value)

    # 文本型和数值型特征
    text_features = ['发酵容器', '制曲原料', '原料', '生产工艺']
    numerical_columns = ['发酵周期', '发酵温度', '用曲量', '制曲温度', '发酵次数', '酒精度', '总酸', '总酯', '固性物']

    # 处理数值特征
    for col in numerical_columns[:5]:
        X_train[col] = X_train[col].apply(parse_range)
        X_test[col] = X_test[col].apply(parse_range)
    for col in numerical_columns[5:]:
        X_train[col] = X_train[col].apply(parse_numeric_with_unit)
        X_test[col] = X_test[col].apply(parse_numeric_with_unit)

    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    X_train_numerical = imputer.fit_transform(X_train[numerical_columns])
    X_test_numerical = imputer.transform(X_test[numerical_columns])

    # 数据标准化
    scaler = MinMaxScaler()
    X_train_numerical_scaled = scaler.fit_transform(X_train_numerical)
    X_test_numerical_scaled = scaler.transform(X_test_numerical)

    # 文本特征编码
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_train_encoded_text = encoder.fit_transform(X_train[text_features])
    X_test_encoded_text = encoder.transform(X_test[text_features])

    # 获取特征名称
    text_feature_names = encoder.get_feature_names_out(text_features)
    all_feature_names = list(text_feature_names) + numerical_columns

    # 合并特征
    X_train_combined = np.hstack((X_train_encoded_text, X_train_numerical))
    X_test_combined = np.hstack((X_test_encoded_text, X_test_numerical))

    # 数据标准化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    # 特征选择（方差过滤）
    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    # 更新特征名称
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [all_feature_names[i] for i in selected_feature_indices]

    return X_train_selected, X_test_selected, y_train_flavor_encoded, y_test_flavor_encoded, selected_feature_names


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    绘制学习曲线
    :param estimator: 模型
    :param title: 标题
    :param X: 特征数据
    :param y: 目标数据
    :param ylim: y轴范围
    :param cv: 交叉验证折数
    :param n_jobs: 并行任务数
    :param train_sizes: 训练样本数量
    :return: 绘制的学习曲线
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练样本数量")
    plt.ylabel("得分")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="训练得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="交叉验证得分")

    plt.legend(loc="best")
    return plt


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    绘制混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param title: 标题
    :param filename: 保存文件名
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    评估模型性能
    :param model: 模型
    :param X_train: 训练集特征
    :param y_train: 训练集目标
    :param X_test: 测试集特征
    :param y_test: 测试集目标
    :return: 评估指标
    """
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 准确率
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # 精确率、召回率、F1值
    train_precision = precision_score(y_train, y_pred_train, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    train_recall = recall_score(y_train, y_pred_train, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    return train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1


def main():
    if not os.path.exists('vis2'):
        os.makedirs('vis2')

    # 读取数据
    data = pd.read_excel('handle_result4.3.xlsx')
    logging.info("数据读取完成")

    # 数据预处理
    X_train_selected, X_test_selected, y_train_flavor_encoded, y_test_flavor_encoded, selected_feature_names = preprocess_data(
        data)
    logging.info("数据预处理完成")

    # 多项式朴素贝叶斯模型
    nb_model = MultinomialNB()  # 初始化多项式朴素贝叶斯模型
    nb_model.fit(X_train_selected, y_train_flavor_encoded)  # 训练模型

    # 绘制学习曲线
    plot_learning_curve(nb_model, "多项式朴素贝叶斯学习曲线", X_train_selected, y_train_flavor_encoded, cv=2)
    plt.savefig('vis2/nb_learning_curve.png')
    plt.close()
    logging.info("学习曲线绘制完成")

    # 评估模型
    train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1 = evaluate_model(
        nb_model, X_train_selected, y_train_flavor_encoded, X_test_selected, y_test_flavor_encoded)
    logging.info("模型评估完成")

    # 可视化准确率对比
    models = ['多项式朴素贝叶斯']
    train_accuracies = [train_accuracy]
    test_accuracies = [test_accuracy]

    plt.figure(figsize=(10, 6))
    bar_width = 0.3
    r1 = np.arange(len(models))

    plt.bar(r1, train_accuracies, width=bar_width, label='训练集准确率', color='blue')
    plt.bar([x + bar_width for x in r1], test_accuracies, width=bar_width, label='测试集准确率', color='lightblue')

    plt.title('多项式朴素贝叶斯模型训练集与测试集准确率对比')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.xticks([r + bar_width / 2 for r in r1], models)
    for i, v in enumerate(train_accuracies):
        plt.text(r1[i], v + 0.01, f'{v:.4f}', ha='center')
    for i, v in enumerate(test_accuracies):
        plt.text(r1[i] + bar_width, v + 0.01, f'{v:.4f}', ha='center')

    plt.legend()
    plt.tight_layout()
    plt.savefig('vis2/nb_accuracy_comparison.png')
    plt.close()
    logging.info("准确率对比图绘制完成")

    # 绘制混淆矩阵
    plot_confusion_matrix(y_test_flavor_encoded, nb_model.predict(X_test_selected), '多项式朴素贝叶斯 - 测试集混淆矩阵',
                          'vis2/nb_confusion_matrix.png')
    logging.info("混淆矩阵绘制完成")

    # 输出所有调试信息和最终指标结果
    logging.info("\n=== 最终模型评估结果 ===")
    logging.info(f"训练集准确率: {train_accuracy:.4f}")
    logging.info(f"测试集准确率: {test_accuracy:.4f}")
    logging.info(f"训练集精确率: {train_precision:.4f}")
    logging.info(f"测试集精确率: {test_precision:.4f}")
    logging.info(f"训练集召回率: {train_recall:.4f}")
    logging.info(f"测试集召回率: {test_recall:.4f}")
    logging.info(f"训练集F1值: {train_f1:.4f}")
    logging.info(f"测试集F1值: {test_f1:.4f}")


if __name__ == "__main__":
    main()
