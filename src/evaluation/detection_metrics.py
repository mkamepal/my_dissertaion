from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def evaluate_detection(trust_scores, malicious_clients, threshold):

    malicious_set = set(malicious_clients)
    y_true = []
    y_pred = []

    for (round_key,clients) in trust_scores.items():
        for (client_id,score) in clients.items():

            # 1 = malicious
            # 0 = benign

            true_label = (1 if client_id in malicious_set else 0)
            predicted_label = (0 if score >= threshold else 1)

            y_true.append(true_label)
            y_pred.append(predicted_label)

    tn, fp, fn, tp = (confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    tpr = (tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    fpr = (fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    return {"accuracy":accuracy_score(y_true, y_pred),
            "precision":precision_score(y_true, y_pred, zero_division=0),
            "recall":recall_score(y_true, y_pred, zero_division=0),
            "f1":f1_score(y_true, y_pred, zero_division=0),
            "TPR":tpr,
            "FPR":fpr,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN":int(fn)
    }