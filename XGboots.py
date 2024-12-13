import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from utils import load_data  # 假设这个文件包含数据加载的函数

# 设置随机种子和基本参数
seed = 42
DATA_PATH = "./v2_datasets/"
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
    silos[_cid]["x_train"] = x_train
    silos[_cid]["y_train"] = y_train
    silos[_cid]["x_test"] = x_test
    silos[_cid]["y_test"] = y_test


# 定义 LSTM 模型
def build_lstm(input_dim):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(input_dim, 1), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # 二分类问题
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 创建一个智能体类，包含训练和预测方法
class Agent:
    def __init__(self, input_dim):
        self.model = build_lstm(input_dim)

    def train(self, x_train, y_train):
        x_train = np.expand_dims(x_train, axis=-1)  # LSTM 需要三维输入 (samples, timesteps, features)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # 早停机制
        self.model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE, verbose=0, validation_split=0.2,
                       callbacks=[early_stopping])

    def predict(self, x_test):
        x_test = np.expand_dims(x_test, axis=-1)
        return (self.model.predict(x_test) > 0.5).astype(int)  # 返回二分类标签（0或1）


# 创建多智能体系统
class MultiAgentSystem:
    def __init__(self, agents):
        self.agents = agents

    def train_agents(self, silos):
        for silo_name, silo_data in silos.items():
            agent = self.agents[silo_name]
            agent.train(silo_data['x_train'], silo_data['y_train'])

    def evaluate_agents(self, silos):
        results = []
        for silo_name, silo_data in silos.items():
            agent = self.agents[silo_name]
            pred = agent.predict(silo_data['x_test'])
            # 评估性能
            results.append(self.evaluate_metrics(silo_data['y_test'], pred))
        return results

    def evaluate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, pre, rec, f1


# 进行消融实验
def ablation_experiment(silos, num_agents_list):
    all_results = {}
    for num_agents in num_agents_list:
        selected_silos = {silo_name: silo_data for silo_name, silo_data in list(silos.items())[:num_agents]}
        agents = {silo_name: Agent(input_dim=silo_data['x_train'].shape[1]) for silo_name, silo_data in
                  selected_silos.items()}
        multi_agent_system = MultiAgentSystem(agents)

        # 训练和评估
        multi_agent_system.train_agents(selected_silos)
        results = multi_agent_system.evaluate_agents(selected_silos)

        # 保存结果
        all_results[num_agents] = results

    return all_results


# 设置消融实验的智能体数量
num_agents_list = [1, 2, 3, 4, len(silos)]  # 从1个智能体到所有智能体

# 运行消融实验
all_results = ablation_experiment(silos, num_agents_list)

# 输出结果并保存到txt文件
with open("ablation_v2_results.txt", "w") as f:
    for num_agents, results in all_results.items():
        f.write(f"Number of agents: {num_agents}\n")
        avg_acc = np.mean([r[0] for r in results])
        avg_pre = np.mean([r[1] for r in results])
        avg_rec = np.mean([r[2] for r in results])
        avg_f1 = np.mean([r[3] for r in results])

        f.write(f"Average Accuracy: {avg_acc:.4f}\n")
        f.write(f"Average Precision: {avg_pre:.4f}\n")
        f.write(f"Average Recall: {avg_rec:.4f}\n")
        f.write(f"Average F1-score: {avg_f1:.4f}\n")
        f.write("\n")

    print(f"Results saved to 'ablation_results.txt'")

