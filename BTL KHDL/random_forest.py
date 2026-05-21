# ==============================
# MODEL - PHÂN TÍCH HÀNH VI MUA HÀNG
# RANDOM FOREST
# ==============================

# ==============================
# 1. IMPORT
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ==============================
# 2. LOAD DATA
# ==============================

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(
        SAVE_DIR,
        "processed_online_shoppers.csv"
    )
)

print("="*55)
print("   PHÂN TÍCH MÔ HÌNH DỰ ĐOÁN MUA HÀNG")
print("="*55)

print(f"\nDataset: {df.shape[0]} dòng, {df.shape[1]} cột")
print(
    f"Tỷ lệ mua hàng: {df['Revenue'].mean()*100:.1f}%"
)

# ==============================
# 3. CHUẨN BỊ DỮ LIỆU
# ==============================

FEATURES = [

'Administrative',
'Administrative_Duration',
'Informational',
'Informational_Duration',
'ProductRelated',
'ProductRelated_Duration',
'BounceRates',
'ExitRates',
'PageValues',
'SpecialDay',
'OperatingSystems',
'Browser',
'Region',
'TrafficType',
'Weekend',
'Month_encoded',
'VisitorType_encoded',
'TotalTime'

]

X=df[FEATURES]

y=df['Revenue']

X_train,X_test,y_train,y_test=\
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 4. RANDOM FOREST
# ==============================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

y_pred = rf.predict(
    X_test
)

y_prob = rf.predict_proba(
    X_test
)[:,1]

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

y_prob=rf.predict_proba(X_test)[:,1]

print("\nTrain:",len(X_train))
print("Test :",len(X_test))


# ==============================
# 4. RANDOM FOREST
# ==============================

rf=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

y_pred=rf.predict(
    X_test
)

y_prob=rf.predict_proba(
    X_test
)[:,1]

# ==============================
# RANDOM FOREST TREE VISUALIZATION
# ==============================

plt.figure(figsize=(22,10))

# lấy cây đầu tiên trong rừng
tree = rf.estimators_[0]

plot_tree(

    tree,

    feature_names=X.columns,

    class_names=[
        'No Purchase',
        'Purchase'
    ],

    filled=True,

    rounded=True,

    max_depth=3,
    fontsize=8
)

plt.title(
    "Random Forest - Tree #1"
)

plt.savefig(

    os.path.join(
        SAVE_DIR,
        "rf_tree.png"
    ),

    dpi=150,
    bbox_inches='tight'
)

plt.show()

print("\n[✓] Đã lưu rf_tree.png")

# ==============================
# 5. ĐÁNH GIÁ
# ==============================

accuracy=accuracy_score(
    y_test,
    y_pred
)

precision=precision_score(
    y_test,
    y_pred
)

recall=recall_score(
    y_test,
    y_pred
)

f1=f1_score(
    y_test,
    y_pred
)

auc=roc_auc_score(
    y_test,
    y_prob
)

print("\n"+"="*55)
print("KẾT QUẢ RANDOM FOREST")
print("="*55)

print(
f"""
Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
F1-score : {f1:.4f}
AUC-ROC  : {auc:.4f}
"""
)

print("\nConfusion Matrix")
print(
confusion_matrix(
    y_test,
    y_pred
)
)

print("\nClassification Report")

print(
classification_report(
    y_test,
    y_pred
)
)


# ==============================
# 6. COMBINE 3 BIỂU ĐỒ
# ==============================

importance = pd.DataFrame({

    'Feature': FEATURES,
    'Importance': rf.feature_importances_

})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

# ==============================
# TOP 10 FEATURE
# ==============================

print("\n"+"="*55)
print("TOP 10 FEATURES QUAN TRỌNG")
print("="*55)

top10 = importance.head(10)

print(
    top10.to_string(
        index=False
    )
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(20,6)
)

fig.suptitle(
    "KẾT QUẢ RANDOM FOREST - DỰ ĐOÁN HÀNH VI MUA HÀNG",
    fontsize=16,
    fontweight='bold'
)

# ==========================
# 1. Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0],
    xticklabels=['Không mua','Mua'],
    yticklabels=['Không mua','Mua']
)

axes[0].set_title(
    "Confusion Matrix",
    fontweight='bold'
)

axes[0].set_xlabel(
    "Dự đoán"
)

axes[0].set_ylabel(
    "Thực tế"
)

# ==========================
# 2. ROC Curve
# ==========================

fpr,tpr,_ = roc_curve(
    y_test,
    y_prob
)

axes[1].plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.3f}"
)

axes[1].plot(
    [0,1],
    [0,1],
    '--'
)

axes[1].set_title(
    "ROC Curve",
    fontweight='bold'
)

axes[1].set_xlabel(
    "False Positive Rate"
)

axes[1].set_ylabel(
    "True Positive Rate"
)

axes[1].legend()

axes[1].grid(alpha=0.3)

# ==========================
# 3. Feature Importance
# ==========================

top10 = importance.head(10)

sns.barplot(
    data=top10,
    x='Importance',
    y='Feature',
    ax=axes[2]
)

axes[2].set_title(
    "Top 10 Features",
    fontweight='bold'
)

axes[2].grid(
    axis='x',
    alpha=0.3
)

# tránh title đè lên biểu đồ
plt.subplots_adjust(
    top=0.82,
    wspace=0.35
)

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "rf_results.png"
    ),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n Đã lưu: rf_results.png")
# ==============================
# 9. KẾT LUẬN
# ==============================

print("\n"+"="*55)
print("KẾT LUẬN")
print("="*55)

print(f"""

Random Forest hoạt động khá tốt:

- Accuracy: {accuracy*100:.2f}%
- F1-score: {f1*100:.2f}%
- AUC: {auc*100:.2f}%

Nhận xét:
- Feature quan trọng nhất:
  {importance.iloc[0]['Feature']}

- Mức ảnh hưởng:
  {importance.iloc[0]['Importance']:.3f}

- Random Forest giảm overfitting
- Dự đoán ổn định
- Recall còn có thể cải thiện
- Có thể thử GridSearchCV hoặc SMOTE

""")

result = pd.DataFrame({

'Model':['Random Forest'],
'Accuracy':[accuracy*100],
'Precision':[precision*100],
'Recall':[recall*100],
'F1':[f1*100],
'AUC':[auc*100]

})

result.to_csv(
    'rf_result.csv',
    index=False
)

# ==============================
# LƯU KẾT QUẢ
# ==============================

result = pd.DataFrame({

'Model':['Random Forest'],

'Accuracy':[accuracy*100],

'Precision':[precision*100],

'Recall':[recall*100],

'F1':[f1*100],

'AUC':[auc*100]

})

result.to_csv(

os.path.join(
SAVE_DIR,
"rf_result.csv"
),

index=False

)

print("\n[✓] Đã lưu rf_result.csv")