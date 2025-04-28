import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if not os.path.exists('vis2'):
    os.makedirs('vis2')

# 1. 读取数据
data = pd.read_excel('handle_result4.3.xlsx')

# 4. 划分数据集
input_columns = ['发酵容器', '制曲原料', '原料', '生产工艺',
                 '发酵周期', '发酵温度', '用曲量', '制曲温度', '发酵次数', '酒精度', '总酸', '总酯', '固性物']
X = data[input_columns]
y_flavor = data['风味']

X_train, X_test, y_train_flavor, y_test_flavor = train_test_split(X, y_flavor, test_size=0.2, random_state=42)

# 5. 处理目标变量（在划分后进行标签化）
unique_flavors = y_train_flavor.unique()
flavor_labels = {flavor: idx for idx, flavor in enumerate(unique_flavors)}

y_train_flavor_encoded = y_train_flavor.map(flavor_labels)
y_test_flavor_encoded = y_test_flavor.map(lambda x: flavor_labels.get(x, -1))


# 6. 处理输入特征
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


text_features = ['发酵容器', '制曲原料', '原料', '生产工艺']
numerical_columns = ['发酵周期', '发酵温度', '用曲量', '制曲温度', '发酵次数', '酒精度', '总酸', '总酯', '固性物']

for col in numerical_columns[:5]:
    X_train[col] = X_train[col].apply(parse_range)
    X_test[col] = X_test[col].apply(parse_range)
for col in numerical_columns[5:]:
    X_train[col] = X_train[col].apply(parse_numeric_with_unit)
    X_test[col] = X_test[col].apply(parse_numeric_with_unit)

imputer = SimpleImputer(strategy='mean')
X_train_numerical = imputer.fit_transform(X_train[numerical_columns])
X_test_numerical = imputer.transform(X_test[numerical_columns])

scaler = MinMaxScaler()
X_train_numerical_scaled = scaler.fit_transform(X_train_numerical)
X_test_numerical_scaled = scaler.transform(X_test_numerical)

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded_text = encoder.fit_transform(X_train[text_features])
X_test_encoded_text = encoder.transform(X_test[text_features])

text_feature_names = encoder.get_feature_names_out(text_features)
all_feature_names = list(text_feature_names) + numerical_columns

X_train_combined = np.hstack((X_train_encoded_text, X_train_numerical))
X_test_combined = np.hstack((X_test_encoded_text, X_test_numerical))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# PCA降维处理
pca = PCA(n_components=0.95)  # 保留95%的方差
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# 定义模型调优函数
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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


# SVM 调优
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

skf = StratifiedKFold(n_splits=5)
svm_grid_search = GridSearchCV(SVC(class_weight='balanced', random_state=42),
                               svm_param_grid, cv=skf, n_jobs=-1)
svm_grid_search.fit(X_train_pca, y_train_flavor_encoded)
best_svm = svm_grid_search.best_estimator_
plot_learning_curve(best_svm, "SVM 学习曲线", X_train_pca, y_train_flavor_encoded, cv=skf)
plt.savefig('vis2/svm_learning_curve.png')
plt.close()

# 训练调优后的模型
best_svm.fit(X_train_pca, y_train_flavor_encoded)

# 预测
y_pred_svm_train = best_svm.predict(X_train_pca)
y_pred_svm_test = best_svm.predict(X_test_pca)

# 交叉验证
cv_scores_svm = cross_val_score(best_svm, X_train_pca, y_train_flavor_encoded, cv=skf)

# 可视化准确率对比
models = ['SVM']
train_accuracies = [
    accuracy_score(y_train_flavor_encoded, y_pred_svm_train)
]
test_accuracies = [
    accuracy_score(y_test_flavor_encoded, y_pred_svm_test)
]

plt.figure(figsize=(10, 6))
bar_width = 0.3
r1 = np.arange(len(models))

plt.bar(r1, train_accuracies, width=bar_width, label='训练集准确率', color='blue')
plt.title('SVM 模型训练集与测试集准确率对比')
plt.ylabel('准确率')
plt.ylim(0, 1)
plt.xticks(r1, models)
for i, v in enumerate(train_accuracies):
    plt.text(r1[i], v + 0.01, f'{v:.4f}', ha='center')

plt.legend()
plt.tight_layout()
plt.savefig('vis2/svm_accuracy_comparison.png')
plt.close()


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig(f'vis2/{filename}')
    plt.close()


# SVM 混淆矩阵
plot_confusion_matrix(y_test_flavor_encoded, y_pred_svm_test, 'SVM - 测试集混淆矩阵', 'svm_confusion_matrix.png')

# 输出所有调试信息和最终指标结果
print("\n=== SVM - 每个类别的性能 ===")
print(classification_report(y_test_flavor_encoded, y_pred_svm_test))

print("\n=== 最终模型评估结果 ===")
print(f"训练集唯一风味数量: {len(unique_flavors)}")

print("\nSVM:")
print(f"训练集准确率: {train_accuracies[0]:.4f}")
print(f"测试集准确率: {test_accuracies[0]:.4f}")
print(f"交叉验证平均准确率: {cv_scores_svm.mean():.4f} (±{cv_scores_svm.std():.4f})")

