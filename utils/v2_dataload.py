import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

seed = 42

# 默认不适用的特征列
not_applicable_features = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Attack', 'Label']


def remove_features(df, full, feats=not_applicable_features):
    # 如果是完整数据集，移除'Dataset'列
    if full:
        feats.remove('Dataset')

    # 去除不适用的特征列，返回特征和标签
    X = df.drop(columns=feats)
    y = df['Label']
    return X, y


def train_test_scaled(X, y, test_size):
    # 使用 MinMaxScaler 对数据进行归一化
    scaler = MinMaxScaler()
    indices = list(X.index)
    X_train, X_test, y_train, y_test, _, test_index = train_test_split(X, y, indices, test_size=test_size,
                                                                       random_state=seed, stratify=y)
    # 对训练集和测试集进行缩放
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, test_index


def load_data(cid, info=True, test_size=0.2, full=False):
    # 判断文件名并加载对应的数据集
    if ("NF-BoT-IoT-v2.csv.gz" in cid or "NF-ToN-IoT-v2.csv.gz" in cid):
        # 读取V2数据集（较大的数据集）
        df = pd.read_csv(cid, low_memory=True, nrows=188937)  # 根据需求设置行数限制
    else:
        # 读取其他数据集（非V2）
        df = pd.read_csv(cid, low_memory=True)

    df.dropna(inplace=True)  # 删除缺失值
    if info:
        # 输出数据集的一些基本信息
        print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}" \
              .format(cid[cid.rfind("/") + 1:cid.find(".csv")], df.shape[0], sum(df.Label == 0), \
                      sum(df.Label == 1), sorted(list(df.Attack.unique().astype(str)))))

    # 移除不适用的特征列
    X, y = remove_features(df, full=full)

    # 划分数据集并进行缩放
    x_train, y_train, x_test, y_test, test_index = train_test_scaled(X, y, test_size)

    if info:
        # 输出错误分析数据（测试集的标签）
        ref = cid[cid.rfind("/") + 1:cid.find(".csv")]
        df['Attack'].iloc[test_index].to_csv("./error_analysis/" + ref + "_test_classes.csv")

    return x_train, y_train, x_test, y_test
