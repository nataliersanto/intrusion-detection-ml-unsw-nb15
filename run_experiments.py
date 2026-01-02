# run_experiments.py
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models

#make folder for results
if not os.path.exists("experiments"):
    os.makedirs("experiments")
col_names = [ #grabbed from official dataset documentation:
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth',
    'res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports',
    'ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label'
]

#load the csvs
csv_files = glob.glob("data/UNSW-NB15_*.csv")
dfs = []
for f in csv_files:
    try:
        df_temp = pd.read_csv(f, low_memory=False)  #try reading with headers
        if 'label' not in df_temp.columns:#if no headers detected...
            #assign official column names
            df_temp = pd.read_csv(f, names=col_names, low_memory=False)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
print(f"Total records loaded: {df.shape[0]}")
df.columns = df.columns.str.strip()  #remove whitespace
print("Columns loaded:", df.columns.tolist())

#the samples have a binary label where 0 is benign and 1 is attack
#categorize the samples as malicious or benign
if 'label' in df.columns:
    df['is_attack'] = df['label'].astype(int) #if binary label is 1, its malicious so add it to the is attack column
else:
    raise ValueError("Column 'label' not found. Check CSV files.")
drop_cols = ['srcip','dstip','proto','service','state','attack_cat','label'] #columns to drop
X = df.drop(columns=drop_cols + ['is_attack']) #"X" for features
y = df['is_attack'] #"y" for target

#convert to numeric and handle non-numeric entries
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

#training and validation
X_train, X_val, y_train, y_val = train_test_split( #80% train, 20% validation
    X, y, test_size=0.2, stratify=y, random_state=42 #keeps same class balance
)
#from StandardScaler documentation:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
#from SMOTE documentation:
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print("Before SMOTE:", y_train.value_counts()) #print before and after to show balancing
print("After SMOTE:", pd.Series(y_train_res).value_counts())
train_times = {} #to store training times for comparative evaluation
models_dict = {}

#logistic regression
start = time.time() #start time
lr = LogisticRegression(max_iter=1000) #ensure convergence
lr.fit(X_train_res, y_train_res)
train_times['Logistic Regression'] = time.time() - start
models_dict['Logistic Regression'] = lr #store model in the dict

#random forest
start = time.time() #start time
rf = RandomForestClassifier(
    n_estimators=200, 
    n_jobs=-1, #use all cores
    random_state=42 #for reproducibility
)
rf.fit(X_train_res, y_train_res)
train_times['Random Forest'] = time.time() - start
models_dict['Random Forest'] = rf #store model in the dict

#xgboost
start = time.time() #start time
xg = xgb.XGBClassifier(
    n_estimators=300, #better performance with more trees
    max_depth=6, #6 to reduce overfitting
    learning_rate=0.1, #how much each tree contributes (each tree corrects previous)
    eval_metric='logloss', #standard metrcic for binary classification
    n_jobs=-1
)
xg.fit(X_train_res, y_train_res)
train_times['XGBoost'] = time.time() - start
models_dict['XGBoost'] = xg #store model in the dict

#neural network
input_dim = X_train_res.shape[1]
nn_model = models.Sequential([ #from tensorflow documentation (the sequential model):
    layers.Dense(128, activation='relu', input_shape=(input_dim,)), #ReLU for non linerarity
    layers.Dropout(0.3), #to reduce overfitting
    layers.Dense(64, activation='relu'), #second hidden layer
    layers.Dense(1, activation='sigmoid') #output layer(0 and 1)
])
nn_model.compile( #also from documentation
    optimizer='adam', #adaptive learning rate optimizer
    loss='binary_crossentropy', #standard for binary classification
    metrics=['AUC'] #track AUC during training
)
start = time.time() #start time
nn_model.fit( #training neural network
    X_train_res, 
    y_train_res,
    epochs=15, 
    batch_size=256,
    validation_data=(X_val_scaled, y_val),
    verbose=1
)
train_times['Neural Network'] = time.time() - start
models_dict['Neural Network'] = nn_model #store model in the dict
print("Training times:", train_times) #print training times for all models

#predicted probability of the attack on validation set
preds_dict = {}
for name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
    preds_dict[name] = models_dict[name].predict_proba(X_val_scaled)[:,1] #returns probability for class 1
preds_dict['Neural Network'] = nn_model.predict(X_val_scaled).flatten() #neural network predict method

#f1 scores for each model
f1_scores = {}
for name, preds in preds_dict.items():
    binary_preds = (preds >= 0.5).astype(int) #default threshold of 0.5
    f1_scores[name] = f1_score(y_val, binary_preds)
print("F1 Scores:", f1_scores)

#ROC curves
plt.figure(figsize=(8,6))
for name, preds in preds_dict.items(): #iterate through each models predictions
    fpr, tpr, _ = roc_curve(y_val, preds) #output false positive rate, true positive rate
    roc_auc = auc(fpr, tpr) #calculate area under curve
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.savefig("experiments/roc_curves.png")
plt.close()

#pr curves
plt.figure(figsize=(8,6))
for name, preds in preds_dict.items():
    precision, recall, _ = precision_recall_curve(y_val, preds) #computes p and r
    plt.plot(recall, precision, label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.savefig("experiments/pr_curves.png")
plt.close()

#confusion matrices
#xgboost
xgb_preds_binary = (preds_dict["XGBoost"] >= 0.5).astype(int)
cm_xgb = confusion_matrix(y_val, xgb_preds_binary)
plt.figure(figsize=(6,5))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (XGBoost)")
plt.savefig("experiments/confusion_matrix_xgb.png")
plt.close()

#random forest
rf_preds_binary = (preds_dict["Random Forest"] >= 0.5).astype(int)
cm_rf = confusion_matrix(y_val, rf_preds_binary)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Random Forest)")
plt.savefig("experiments/confusion_matrix_rf.png")
plt.close()

#bar chart of F1 scores
plt.figure(figsize=(8,6))
plt.bar(f1_scores.keys(), f1_scores.values())
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison")
plt.ylim(0,1)
plt.grid(axis='y')
plt.savefig("experiments/f1_scores.png")
plt.close()
