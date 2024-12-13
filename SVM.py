from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
from utils import load_data
from sklearn.svm import OneClassSVM

# 设置随机种子和基本参数
seed = 42
DATA_PATH = "./v1_datasets/"
BATCH_SIZE = 128

# 加载数据集
cids = []
for _, _, cid in os.walk(DATA_PATH):
    cids.extend(cid)

silos = {}

for cid in cids:
    _cid = cid[:cid.find(".csv")]
    silos[_cid] = {}
    x_train, y_train, x_test, y_test = load_data.load_data(os.path.join(DATA_PATH, cid), info=False)

    # 不再使用提示特征
    silos[_cid]["x_train"] = x_train
    silos[_cid]["y_train"] = y_train
    silos[_cid]["x_test"] = x_test
    silos[_cid]["y_test"] = y_test

# 定义 One-Class SVM 模型
ocsvm_model = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')

# 评估 One-Class SVM 模型
local_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

for silo_name, silo_data in silos.items():
    print(f"> Evaluating One-Class SVM on Silo: {silo_name}")

    # 训练 One-Class SVM 模型，仅使用正常数据（y_train == 0）
    ocsvm_model.fit(silo_data["x_train"][silo_data["y_train"] == 0])

    # 本地测试集评估
    y_test = silo_data["y_test"].map({0: 1, 1: -1})  # One-Class SVM 返回 +1 和 -1
    pred = ocsvm_model.predict(silo_data["x_test"])

    # 计算本地评估指标
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, pos_label=1)
    rec = recall_score(y_test, pred, pos_label=1)
    f1 = f1_score(y_test, pred, pos_label=1)

    local_metrics["accuracy"].append(acc)
    local_metrics["precision"].append(pre)
    local_metrics["recall"].append(rec)
    local_metrics["f1"].append(f1)

    print(f">> Local Metrics on {silo_name}: ACC={acc:.4f}, PRE={pre:.4f}, REC={rec:.4f}, F1={f1:.4f}")

# 输出平均结果
print(f">> Average Local Accuracy: {np.mean(local_metrics['accuracy']):.4f} ± {np.std(local_metrics['accuracy']):.4f}")
print(f">> Average Local Precision: {np.mean(local_metrics['precision']):.4f} ± {np.std(local_metrics['precision']):.4f}")
print(f">> Average Local Recall: {np.mean(local_metrics['recall']):.4f} ± {np.std(local_metrics['recall']):.4f}")
print(f">> Average Local F1-score: {np.mean(local_metrics['f1']):.4f} ± {np.std(local_metrics['f1']):.4f}")
