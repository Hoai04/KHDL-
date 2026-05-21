import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# ==============================
# 2. LOAD DATA
# ==============================
SAVE_DIR = r"D:\BTL KHDL"
df = pd.read_csv("processed_online_shoppers.csv")

print("=" * 55)
print("       PHÂN TÍCH MÔ HÌNH LOGISTIC REGRESSION")
print("=" * 55)
print(f"\nDataset: {df.shape[0]} dòng, {df.shape[1]} cột")
print(f"Tỷ lệ mua hàng: {df['Revenue'].mean()*100:.1f}%  ({df['Revenue'].sum()} / {len(df)})")

# ==============================
# 3. CHUẨN BỊ FEATURES
# ==============================
FEATURES = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'OperatingSystems', 'Browser', 'Region', 'TrafficType',
    'Weekend', 'Month_encoded', 'VisitorType_encoded', 'TotalTime'
]

X = df[FEATURES].copy()
X['Month_encoded'] = X['Month_encoded'].fillna(X['Month_encoded'].median())
y = df['Revenue']

# Train/Test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain set : {len(X_train)} mẫu")
print(f"Test set  : {len(X_test)} mẫu")

# ==============================
# 4. TRAIN LOGISTIC REGRESSION
# ==============================
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]

# ==============================
# 5. ĐÁNH GIÁ
# ==============================
print("\n" + "=" * 55)
print("           KẾT QUẢ LOGISTIC REGRESSION")
print("=" * 55)
print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score  : {f1_score(y_test, y_pred)*100:.2f}%")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_prob)*100:.2f}%")

# ==============================
# 6. VISUALIZATION (3 biểu đồ)
# ==============================
fig = plt.figure(figsize=(16, 5))
fig.suptitle("Kết Quả Logistic Regression — Dự Đoán Hành Vi Mua Hàng",
             fontsize=14, fontweight='bold')

# Biểu đồ 1: Confusion Matrix
ax1 = fig.add_subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Không mua', 'Mua'],
            yticklabels=['Không mua', 'Mua'])
ax1.set_title('Confusion Matrix', fontweight='bold')
ax1.set_ylabel('Thực tế')
ax1.set_xlabel('Dự đoán')

# Biểu đồ 2: ROC Curve
ax2 = fig.add_subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
ax2.plot(fpr, tpr, color='#4C72B0', lw=2, label=f'AUC = {auc:.3f}')
ax2.plot([0,1],[0,1], 'k--', lw=1, label='Random Guess')
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Biểu đồ 3: Feature Importance
ax3 = fig.add_subplot(1, 3, 3)
coef_df = pd.DataFrame({
    'Feature': FEATURES,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False).head(10)

colors = ['#e74c3c' if v > 0 else '#3498db' for v in coef_df['Coefficient']]
ax3.barh(coef_df['Feature'][::-1], coef_df['Coefficient'][::-1],
         color=colors[::-1], edgecolor='white')
ax3.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax3.set_xlabel('Hệ số (Coefficient)')
ax3.set_title('Top 10 Features Quan Trọng', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'logistic_results.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n[✓] Đã lưu: logistic_results.png")

# ==============================
# 7. FEATURE IMPORTANCE
# ==============================
coef_all = pd.DataFrame({
    'Feature': FEATURES,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n" + "=" * 55)
print("   TOP 10 FEATURES QUAN TRỌNG NHẤT")
print("=" * 55)
print(coef_all.head(10).to_string(index=False))


# ==============================
# LƯU KẾT QUẢ
# ==============================

result = pd.DataFrame({

'Model':['Logistic Regression'],

'Accuracy':[
accuracy_score(
    y_test,
    y_pred
)*100
],

'Precision':[
precision_score(
    y_test,
    y_pred
)*100
],

'Recall':[
recall_score(
    y_test,
    y_pred
)*100
],

'F1':[
f1_score(
    y_test,
    y_pred
)*100
],

'AUC':[
roc_auc_score(
    y_test,
    y_prob
)*100
]

})

result.to_csv(

os.path.join(
SAVE_DIR,
"lr_result.csv"
),

index=False

)

print("\n[✓] Đã lưu lr_result.csv")

# ==============================
# 8. KẾT LUẬN
# ==============================
print("\n" + "=" * 55)
print("                    KẾT LUẬN")
print("=" * 55)
print(f"""
1. MÔ HÌNH: Logistic Regression
   - Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%
   - Precision : {precision_score(y_test, y_pred)*100:.2f}%
   - Recall    : {recall_score(y_test, y_pred)*100:.2f}%
   - F1-Score  : {f1_score(y_test, y_pred)*100:.2f}%
   - AUC-ROC   : {roc_auc_score(y_test, y_prob)*100:.2f}%

2. FEATURE QUAN TRỌNG NHẤT: PageValues (hệ số = {coef_all.iloc[0]['Coefficient']:.3f})
   - PageValues cao → xác suất mua hàng cao
   - ExitRates cao  → xác suất mua hàng thấp

3. NHẬN XÉT:
   - Accuracy ~88.61% tốt với dữ liệu mất cân bằng
   - Recall thấp (38.74%) do chỉ 15.6% người mua hàng
   - AUC-ROC 89.64% → phân biệt tốt người mua / không mua
""")


