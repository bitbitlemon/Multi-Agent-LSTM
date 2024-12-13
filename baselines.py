import os
import numpy as np
from utils import model, load_data
from efc import EnergyBasedFlowClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def eval_dae(x, x_hat, threshold):
    losses = np.mean(abs(x - x_hat), axis=1)
    pred = losses > threshold
    return 1*pred


seed = 42
DATA_PATH = "./full_datasets/"
BATCH_SIZE = 128
EPOCHS = 10

cids = []
for _, _, cid in os.walk(DATA_PATH):
    cids.extend(cid)

silos = {}

for cid in cids:
    _cid = cid[:cid.find(".csv")]
    silos[_cid] = {}
    x_train, y_train, x_test, y_test = load_data.load_data(DATA_PATH + cid, info=False)
    silos[_cid]["x_train"] = x_train
    silos[_cid]["y_train"] = y_train
    silos[_cid]["x_test"] = x_test
    silos[_cid]["y_test"] = y_test

dae = model.create_model(x_train.shape[1])

models = {
        "IsoForest" : { "model" : IsolationForest(n_jobs=-1, random_state=seed), "local_perf" : [], "cross_perf" : []},
        #"OneClassSVM" : { "model" : OneClassSVM(), "local_perf" : [], "cross_perf" : []},
        "LOF" : {"model" : LocalOutlierFactor(n_jobs=-1, novelty=True), "local_perf" : [], "cross_perf" : []},
        "DeepAE" : {"model" : dae, "local_perf" : [], "cross_perf" :[] },
        "EFC" : { "model" : EnergyBasedFlowClassifier(cutoff_quantile=0.95), "local_perf" : [], "cross_perf" : []},
        }

results_file = "evaluation_results.txt"
with open(results_file, "w") as result_file:
    # autoencoder = model.create_model(self.x_train.shape[1])

    for algo, model in models.items():
        result_file.write("> Evaluating {}\n".format(algo))
        for silo_name, silo_data in silos.items():
            if algo == "EFC":
                model["model"].fit(silo_data["x_train"], silo_data["y_train"], base_class=0) # EFC requires fit(X,y)
                y_test = silo_data["y_test"]
            else:
                if algo == "DeepAE":
                    idx = int(silo_data["x_train"][silo_data["y_train"]==0].shape[0] * 0.9)
                    train_data = silo_data["x_train"][silo_data["y_train"]==0]
                    val_data = train_data[idx:]
                    train_data = train_data[:idx]

                    history = model["model"].fit(
                            train_data, 
                            train_data,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            shuffle=True
                            )

                    losses = np.mean(np.abs(val_data - model["model"].predict(val_data)), axis=1) # reconstruction MAE on validation set
                    threshold = np.quantile(losses, 0.95) # set an appropriate threshold
                    y_test = silo_data["y_test"]
                else:
                    model["model"].fit(silo_data["x_train"][silo_data["y_train"]==0]) # train each model on benign data
                    y_test = silo_data["y_test"].map({0: 1, 1: -1}) # LOF, OneSVM and IF predict methods returns +1 or -1

            pred = model["model"].predict((silo_data["x_test"]))

            if algo == "DeepAE":
                pred = eval_dae(silo_data["x_test"], pred, threshold)
            
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, average='binary')
            recall = recall_score(y_test, pred, average='binary')
            f1 = f1_score(y_test, pred)

            result_file.write(">> LOCALIZED >> Algo: {} | Silo: {} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1-Score: {:.4f}\n".format(
                algo, silo_name, accuracy, precision, recall, f1))

            model["local_perf"].append(f1) # local performance

            for cross_silo_name, cross_silo_data in silos.items():
                if silo_name != cross_silo_name:
                    y_test_cross = cross_silo_data["y_test"]
                    
                    if algo != "EFC" and algo != "DeepAE":
                        y_test_cross = cross_silo_data["y_test"].map({0:1, 1:-1})

                    x_test_cross = cross_silo_data["x_test"]
                    pred_cross = model["model"].predict(x_test_cross)

                    if algo == "DeepAE":
                        dae_pred_cross = pred_cross
                        pred_cross = eval_dae(x_test_cross, dae_pred_cross, threshold)
                    
                    accuracy_cross = accuracy_score(y_test_cross, pred_cross)
                    precision_cross = precision_score(y_test_cross, pred_cross, average='binary')
                    recall_cross = recall_score(y_test_cross, pred_cross, average='binary')
                    f1_cross = f1_score(y_test_cross, pred_cross)

                    result_file.write("Trained on {} evaluated on {}: Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1-Score: {:.4f}\n".format(
                        silo_name, cross_silo_name, accuracy_cross, precision_cross, recall_cross, f1_cross))

                    model["cross_perf"].append(f1_cross) # cross silo performance

        result_file.write(">> Average Local F1-score: {:.4f} \u00B1 {:.4f}\n".format(np.mean(model["local_perf"]), np.std(model["local_perf"])))
        result_file.write(">> Average Cross F1-score: {:.4f} \u00B1 {:.4f}\n".format(np.mean(model["cross_perf"]), np.std(model["cross_perf"])))
