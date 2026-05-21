# ==========================================
# DECISION TREE - ONLINE SHOPPERS
# ==========================================

# ==============================
# 1. IMPORT
# ==============================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

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

SAVE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

df = pd.read_csv(

    os.path.join(
        SAVE_DIR,
        "processed_online_shoppers.csv"
    )
)

print("="*55)
print("       DECISION TREE")
print("="*55)

print(
    f"\nDataset: {df.shape[0]} dòng"
)

print(
    f"Số cột: {df.shape[1]}"
)

print(
f"Tỷ lệ mua hàng: {df['Revenue'].mean()*100:.1f}%"
)

# ==============================
# 3. FEATURE ENGINEERING
# ==============================

df['TotalTime2']=(
    df['Administrative_Duration']
    +df['Informational_Duration']
    +df['ProductRelated_Duration']
)

# bỏ cột text

df=df.drop(
    columns=['Month','VisitorType']
)

# ==============================
# 4. PREPARE DATA
# ==============================

X=df.drop(
    'Revenue',
    axis=1
)

y=df['Revenue']

X_train,X_test,y_train,y_test=\
train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain:",len(X_train))
print("Test :",len(X_test))

# ==============================
# 5. MODEL
# ==============================

model=DecisionTreeClassifier(

    criterion='entropy',
    max_depth=3,
    min_samples_leaf=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel trained!")

# ==============================
# 6. PREDICT
# ==============================

y_pred=model.predict(
    X_test
)

y_prob=model.predict_proba(
    X_test
)[:,1]

# ==============================
# 7. EVALUATION
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
print("KẾT QUẢ")
print("="*55)

print(f"""
Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
F1-score : {f1:.4f}
AUC      : {auc:.4f}
""")

print(
classification_report(
    y_test,
    y_pred
)
)

# ==============================
# 8. FEATURE IMPORTANCE
# ==============================

importance=pd.DataFrame({

'Feature':X.columns,

'Importance':
model.feature_importances_

})

importance=importance.sort_values(

    by='Importance',
    ascending=False
)

print("\n"+"="*55)
print("TOP 10 FEATURE")
print("="*55)

print(
importance.head(10)
)

# ==============================
# 9. GỘP 3 BIỂU ĐỒ
# ==============================

fig=plt.figure(
    figsize=(18,6)
)

fig.suptitle(

"DECISION TREE RESULTS",

fontsize=15,
fontweight='bold'
)

# ----------------
# Confusion Matrix
# ----------------

ax1=fig.add_subplot(
    1,3,1
)

cm=confusion_matrix(
    y_test,
    y_pred
)

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    ax=ax1,

    xticklabels=[
    'Không mua',
    'Mua'
    ],

    yticklabels=[
    'Không mua',
    'Mua'
    ]
)

ax1.set_title(
    'Confusion Matrix'
)

# ----------------
# ROC
# ----------------

ax2=fig.add_subplot(
    1,3,2
)

fpr,tpr,_=roc_curve(
    y_test,
    y_prob
)

ax2.plot(

    fpr,
    tpr,

    label=f"AUC={auc:.3f}"
)

ax2.plot(
    [0,1],
    [0,1],
    '--'
)

ax2.legend()

ax2.grid()

ax2.set_title(
    'ROC Curve'
)

# ----------------
# TOP FEATURE
# ----------------

ax3=fig.add_subplot(
    1,3,3
)

top10=importance.head(10)

sns.barplot(

    data=top10,

    x='Importance',

    y='Feature',

    ax=ax3
)

ax3.set_title(
    'Top Feature'
)

plt.tight_layout()

plt.savefig(

os.path.join(
SAVE_DIR,
"dt_results.png"
),

dpi=150
)

plt.show()

print(
"\n[✓] Đã lưu dt_results.png"
)

# ==============================
# 10. VẼ CÂY
# ==============================

plt.figure(
    figsize=(20,10)
)

plot_tree(

    model,

    feature_names=X.columns,

    class_names=[
        'No Purchase',
        'Purchase'
    ],

    filled=True,

    rounded=True,

    fontsize=9
)

plt.title(
    "Decision Tree Visualization"
)

plt.savefig(

os.path.join(

SAVE_DIR,

"decision_tree_visualization.png"

),

dpi=150,
bbox_inches='tight'
)

plt.show()

print(
"[✓] Đã lưu decision_tree_visualization.png"
)

# ==============================
# 11. LƯU CSV
# ==============================

result=pd.DataFrame({

'Model':
['Decision Tree'],

'Accuracy':
[accuracy*100],

'Precision':
[precision*100],

'Recall':
[recall*100],

'F1':
[f1*100],

'AUC':
[auc*100]

})

result.to_csv(

os.path.join(

SAVE_DIR,

"dt_result.csv"

),

index=False
)

print(
"\n[✓] Đã lưu dt_result.csv"
)

# ==============================
# 12. KẾT LUẬN
# ==============================

print("\n"+"="*55)
print("KẾT LUẬN")
print("="*55)

print(f"""

Decision Tree:

Accuracy : {accuracy*100:.2f}%
F1-score : {f1*100:.2f}%
AUC      : {auc*100:.2f}%

Nhận xét:

- Dễ giải thích
- Có thể overfitting
- Feature mạnh nhất thường là:
  PageValues
- Có thể cải thiện bằng:
  Random Forest

""")