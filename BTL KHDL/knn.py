# ==================================================
# LDA + KNN (THỬ NHIỀU K)
# ==================================================

import os
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)
SAVE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ==================================================
# 1. Đọc dữ liệu
# ==================================================

data = pd.read_csv("processed_online_shoppers.csv")

# ==================================================
# 2. Encode Month
# ==================================================

if 'Month_encoded' in data.columns:
    data = data.drop('Month_encoded', axis=1)

month_encoder = LabelEncoder()

data['Month_encoded'] = month_encoder.fit_transform(data['Month'])

# ==================================================
# 3. Encode VisitorType
# ==================================================

if 'VisitorType_encoded' not in data.columns:

    visitor_encoder = LabelEncoder()

    data['VisitorType_encoded'] = visitor_encoder.fit_transform(
        data['VisitorType']
    )

# ==================================================
# 4. Bỏ cột text
# ==================================================

data = data.drop(['Month', 'VisitorType'], axis=1)

# ==================================================
# 5. Chia X và y
# ==================================================

X = data.drop('Revenue', axis=1)
y = data['Revenue']

# ==================================================
# 6. Train/Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==================================================
# 7. Scaling
# ==================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 8. LDA
# ==================================================

# Vì dataset có 2 class:
# Revenue = 0 và Revenue = 1
# nên LDA tối đa chỉ còn 1 chiều

lda = LinearDiscriminantAnalysis(n_components=1)

X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

print("=" * 60)
print("LDA SHAPE")
print("=" * 60)

print("Original shape :", X_train_scaled.shape)
print("After LDA      :", X_train_lda.shape)

# ==================================================
# TRAIN FINAL MODEL (K=127)
# ==================================================

print("\n"+"="*60)
print("FINAL MODEL: K = 127")
print("="*60)

knn = KNeighborsClassifier(
    n_neighbors=127,
    algorithm='kd_tree'
)

knn.fit(
    X_train_lda,
    y_train
)

y_pred = knn.predict(
    X_test_lda
)

y_prob = knn.predict_proba(
    X_test_lda
)[:,1]

# ==================================================
# METRICS
# ==================================================

from sklearn.metrics import roc_auc_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")


# ==============================
# LƯU KẾT QUẢ
# ==============================

result = pd.DataFrame({

'Model':['KNN (K=127)'],

'Accuracy':[accuracy*100],

'Precision':[precision*100],

'Recall':[recall*100],

'F1':[f1*100],

'AUC':[auc*100]

})

result.to_csv(

os.path.join(
    SAVE_DIR,
    "knn_result.csv"
),

index=False

)

print("\n[✓] Đã lưu knn_result.csv")