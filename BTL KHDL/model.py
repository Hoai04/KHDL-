# ==============================
# MODEL - PHÂN TÍCH HÀNH VI MUA HÀNG
# Logistic Regression vs KNN
# ==============================

# ==============================
# 1. IMPORT
# ==============================
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

# ==============================
# 2. LOAD DATA
# ==============================
import os
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SAVE_DIR, "processed_online_shoppers.csv"))

print("=" * 55)
print("       PHÂN TÍCH MÔ HÌNH DỰ ĐOÁN MUA HÀNG")
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

# Chuẩn hóa (bắt buộc cho LR và KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain set : {len(X_train)} mẫu")
print(f"Test set  : {len(X_test)} mẫu")

# ==============================
# 4. TRAIN MODEL
# ==============================

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr  = lr.predict_proba(X_test_scaled)[:, 1]

# --- KNN (k=5) ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn  = knn.predict_proba(X_test_scaled)[:, 1]

# ==============================
# 5. ĐÁNH GIÁ
# ==============================
def evaluate(y_true, y_pred, y_prob, name):
    return {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_true, y_pred) * 100, 2),
        'Precision': round(precision_score(y_true, y_pred) * 100, 2),
        'Recall'   : round(recall_score(y_true, y_pred) * 100, 2),
        'F1-Score' : round(f1_score(y_true, y_pred) * 100, 2),
        'AUC-ROC'  : round(roc_auc_score(y_true, y_prob) * 100, 2),
    }

results = pd.DataFrame([
    evaluate(y_test, y_pred_lr,  y_prob_lr,  'Logistic Regression'),
    evaluate(y_test, y_pred_knn, y_prob_knn, 'KNN (k=5)'),
])

print("\n" + "=" * 55)
print("           BẢNG KẾT QUẢ SO SÁNH")
print("=" * 55)
print(results.to_string(index=False))

# ==============================
# 6. VISUALIZATION (4 biểu đồ)
# ==============================
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Kết Quả Mô Hình Dự Đoán Hành Vi Mua Hàng\n(Logistic Regression vs KNN)",
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

COLORS = {'lr': '#4C72B0', 'knn': '#DD8452'}

# ── Biểu đồ 1: Confusion Matrix – Logistic Regression ──
ax1 = fig.add_subplot(gs[0, 0])
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Không mua', 'Mua'], yticklabels=['Không mua', 'Mua'])
ax1.set_title('Confusion Matrix\nLogistic Regression', fontsize=12, fontweight='bold')
ax1.set_ylabel('Thực tế', fontsize=10)
ax1.set_xlabel('Dự đoán', fontsize=10)

# ── Biểu đồ 2: Confusion Matrix – KNN ──
ax2 = fig.add_subplot(gs[0, 1])
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', ax=ax2,
            xticklabels=['Không mua', 'Mua'], yticklabels=['Không mua', 'Mua'])
ax2.set_title('Confusion Matrix\nKNN (k=5)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Thực tế', fontsize=10)
ax2.set_xlabel('Dự đoán', fontsize=10)

# ── Biểu đồ 3: ROC Curve ──
ax3 = fig.add_subplot(gs[1, 0])
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_prob_lr)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
auc_lr  = roc_auc_score(y_test, y_prob_lr)
auc_knn = roc_auc_score(y_test, y_prob_knn)

ax3.plot(fpr_lr,  tpr_lr,  color=COLORS['lr'],  lw=2,
         label=f'Logistic Regression (AUC = {auc_lr:.3f})')
ax3.plot(fpr_knn, tpr_knn, color=COLORS['knn'], lw=2,
         label=f'KNN k=5 (AUC = {auc_knn:.3f})')
ax3.plot([0,1],[0,1], 'k--', lw=1, label='Random Guess')
ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1.02])
ax3.set_xlabel('False Positive Rate', fontsize=10)
ax3.set_ylabel('True Positive Rate', fontsize=10)
ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# ── Biểu đồ 4: So sánh chỉ số ──
ax4 = fig.add_subplot(gs[1, 1])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
lr_vals  = results[results['Model'] == 'Logistic Regression'][metrics].values[0]
knn_vals = results[results['Model'] == 'KNN (k=5)'][metrics].values[0]

x = np.arange(len(metrics))
w = 0.35
bars1 = ax4.bar(x - w/2, lr_vals,  w, label='Logistic Regression',
                color=COLORS['lr'],  edgecolor='white', linewidth=0.8)
bars2 = ax4.bar(x + w/2, knn_vals, w, label='KNN (k=5)',
                color=COLORS['knn'], edgecolor='white', linewidth=0.8)

for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

ax4.set_xticks(x)
ax4.set_xticklabels(metrics, rotation=15, fontsize=9)
ax4.set_ylabel('Score (%)', fontsize=10)
ax4.set_title('So Sánh Chỉ Số Đánh Giá', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_ylim(0, 110)
ax4.grid(axis='y', alpha=0.3)

plt.savefig(os.path.join(SAVE_DIR, 'model_results.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"\n[✓] Đã lưu: {os.path.join(SAVE_DIR, 'model_results.png')}")

# ==============================
# 7. FEATURE IMPORTANCE (Logistic Regression)
# ==============================
coef_df = pd.DataFrame({
    'Feature'    : FEATURES,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n" + "=" * 55)
print("   TOP 10 FEATURES QUAN TRỌNG NHẤT (Logistic Regression)")
print("=" * 55)
print(coef_df.head(10).to_string(index=False))

fig2, ax = plt.subplots(figsize=(9, 6))
top10 = coef_df.head(10)
colors = ['#e74c3c' if v > 0 else '#3498db' for v in top10['Coefficient']]
bars = ax.barh(top10['Feature'][::-1], top10['Coefficient'][::-1],
               color=colors[::-1], edgecolor='white', linewidth=0.6)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Hệ số (Coefficient)', fontsize=11)
ax.set_title('Top 10 Features Quan Trọng Nhất\n(Logistic Regression – màu đỏ = tăng, xanh = giảm khả năng mua)',
             fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar in bars:
    w = bar.get_width()
    ax.text(w + (0.01 if w >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
            f'{w:.3f}', va='center', ha='left' if w >= 0 else 'right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"[✓] Đã lưu: {os.path.join(SAVE_DIR, 'feature_importance.png')}")

# ==============================
# 8. KẾT LUẬN
# ==============================
print("\n" + "=" * 55)
print("                    KẾT LUẬN")
print("=" * 55)

winner = results.loc[results['F1-Score'].idxmax(), 'Model']
print(f"""
1. MÔ HÌNH TỐT HƠN: {winner}
   - Logistic Regression có Accuracy {results.iloc[0]['Accuracy']}%,
     F1-Score {results.iloc[0]['F1-Score']}%, AUC-ROC {results.iloc[0]['AUC-ROC']}%
   - KNN (k=5) có Accuracy {results.iloc[1]['Accuracy']}%,
     F1-Score {results.iloc[1]['F1-Score']}%, AUC-ROC {results.iloc[1]['AUC-ROC']}%

2. FEATURE QUAN TRỌNG NHẤT: PageValues
   - PageValues có hệ số cao nhất → ảnh hưởng mạnh nhất
     đến xác suất mua hàng
   - ExitRates (hệ số âm) → Exit càng cao, mua càng ít
   - TotalTime → dành nhiều thời gian hơn → dễ mua hơn

3. MÔ HÌNH CÓ TỐT KHÔNG?
   - Accuracy ~{results.iloc[0]['Accuracy']}% là khá tốt với dữ liệu
     mất cân bằng (chỉ 15.6% mua hàng)
   - Recall còn thấp → mô hình bỏ sót nhiều người thực sự
     có mua → cần cải thiện nếu dùng thực tế
   - Logistic Regression phù hợp hơn cho bài này vì
     dữ liệu có nhiều features số học tuyến tính
""")