import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from utils import load_data

# Set random seed and basic parameters
seed = 42
DATA_PATH = "./v1_datasets/"
BATCH_SIZE = 128

# Load datasets
cids = []
for _, _, cid in os.walk(DATA_PATH):
    cids.extend(cid)

silos = {}

for cid in cids:
    _cid = cid[:cid.find(".csv")]
    silos[_cid] = {}
    x_train, y_train, x_test, y_test = load_data.load_data(os.path.join(DATA_PATH, cid), info=False)

    # Use the original training and test data directly (without adding prompt features)
    silos[_cid]["x_train"] = x_train
    silos[_cid]["y_train"] = y_train
    silos[_cid]["x_test"] = x_test
    silos[_cid]["y_test"] = y_test


# Define the LSTM model
def build_lstm(input_dim):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(input_dim, 1), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification problem
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create an agent class with training and prediction methods
class Agent:
    def __init__(self, input_dim):
        self.model = build_lstm(input_dim)

    def train(self, x_train, y_train):
        x_train = np.expand_dims(x_train, axis=-1)  # LSTM requires 3D input (samples, timesteps, features)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Early stopping mechanism
        self.model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE, verbose=0, validation_split=0.2,
                       callbacks=[early_stopping])

    def predict(self, x_test):
        x_test = np.expand_dims(x_test, axis=-1)
        return (self.model.predict(x_test) > 0.5).astype(int)  # Return binary labels (0 or 1)


# Create a multi-agent system
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
            # Evaluate performance
            results.append(self.evaluate_metrics(silo_data['y_test'], pred))
        return results

    def evaluate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, pre, rec, f1


# Initialize the multi-agent system
agents = {silo_name: Agent(input_dim=silo_data['x_train'].shape[1]) for silo_name, silo_data in silos.items()}
multi_agent_system = MultiAgentSystem(agents)

# Train and evaluate
multi_agent_system.train_agents(silos)
results = multi_agent_system.evaluate_agents(silos)


with open("v1_results.txt", "w") as f:
    for silo_name, (acc, pre, rec, f1) in zip(silos.keys(), results):
        f.write(f"Silo: {silo_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {pre:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write("\n")

    avg_acc = np.mean([r[0] for r in results])  
    avg_pre = np.mean([r[1] for r in results])
    avg_rec = np.mean([r[2] for r in results])
    avg_f1 = np.mean([r[3] for r in results])

    f.write(f"Average Accuracy: {avg_acc:.4f}\n")
    f.write(f"Average Precision: {avg_pre:.4f}\n")
    f.write(f"Average Recall: {avg_rec:.4f}\n")
    f.write(f"Average F1-score: {avg_f1:.4f}\n")
    print(f"Average F1-score: {avg_f1:.4f}\n")
    print(f"Average Accuracy: {avg_acc:.4f}\n")

print("Evaluation results saved to 'v2_results.txt'")
