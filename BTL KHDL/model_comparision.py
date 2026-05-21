# =====================================================
# ONLINE SHOPPERS - DATA PREPROCESSING
# Môn: Nhập môn Khoa học Dữ liệu
# =====================================================

# =========================
# 1. IMPORT
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import os
from sklearn.preprocessing import LabelEncoder

plt.style.use("seaborn-v0_8")

# =========================
# 2. DOWNLOAD DATA
# =========================

print("="*55)
print("TẢI DỮ LIỆU")
print("="*55)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"

file_name = "online_shoppers.csv"

if not os.path.exists(file_name):

    print("Đang tải dataset...")

    urllib.request.urlretrieve(
        url,
        file_name
    )

    print("Đã tải xong!")

else:

    print("Dataset đã tồn tại.")

# =========================
# 3. LOAD DATA
# =========================

df = pd.read_csv(file_name)

print("\nDataset:")
print(df.shape)

print("\n5 dòng đầu:")
print(df.head())

# =========================
# 4. KIỂM TRA DỮ LIỆU
# =========================

print("\n"+"="*55)
print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
print("="*55)

print("\nMissing Values:")

print(
    df.isnull().sum()
)

duplicates=df.duplicated().sum()

print("\nDuplicate rows:",duplicates)

if duplicates>0:

    df=df.drop_duplicates()

    print(
        "Đã xóa dòng trùng"
    )

print("\nDataset mới:")

print(df.shape)

# =========================
# 5. FEATURE ENGINEERING
# =========================

print("\n"+"="*55)
print("FEATURE ENGINEERING")
print("="*55)

# Boolean → int

df['Weekend']=\
df['Weekend'].astype(int)

df['Revenue']=\
df['Revenue'].astype(int)

# Month encoding

month_map={

'Jan':1,
'Feb':2,
'Mar':3,
'Apr':4,
'May':5,
'June':6,
'Jul':7,
'Aug':8,
'Sep':9,
'Oct':10,
'Nov':11,
'Dec':12

}

df['Month_encoded']=\
df['Month'].map(
    month_map
)

# Visitor Type encoding

le=LabelEncoder()

df['VisitorType_encoded']=\
le.fit_transform(
    df['VisitorType']
)

# Total time

df['TotalTime']=(
df['Administrative_Duration']
+
df['Informational_Duration']
+
df['ProductRelated_Duration']
)

print("Đã tạo:")

print("- Month_encoded")
print("- VisitorType_encoded")
print("- TotalTime")

# =========================
# 6. THÔNG TIN DATA
# =========================

print("\n"+"="*55)
print("THÔNG TIN DATA")
print("="*55)

print("\nSố dòng:",df.shape[0])

print("Số cột:",df.shape[1])

print(
"\nTỷ lệ mua hàng:"
)

purchase_rate=\
df['Revenue'].mean()*100

print(
f"{purchase_rate:.2f}%"
)

print("\nKiểu dữ liệu:")

print(
df.dtypes
)

# =========================
# 7. LƯU DATA
# =========================

save_file=\
"processed_online_shoppers.csv"

df.to_csv(
save_file,
index=False
)

print("\nĐã lưu:")

print(save_file)

# =========================
# 8. VISUALIZATION
# =========================

print("\nTạo biểu đồ...")

fig=plt.figure(
    figsize=(15,5)
)

fig.suptitle(
"ONLINE SHOPPERS DATA OVERVIEW",
fontsize=15,
fontweight='bold'
)

# ------------------
# Revenue
# ------------------

ax1=fig.add_subplot(1,3,1)

sns.countplot(
    data=df,
    x='Revenue',
    ax=ax1
)

ax1.set_title(
"Revenue Distribution"
)

# ------------------
# VisitorType
# ------------------

ax2=fig.add_subplot(1,3,2)

sns.barplot(
    data=df,
    x='VisitorType',
    y='Revenue',
    ax=ax2
)

ax2.set_title(
"Visitor Type"
)

# ------------------
# Heatmap
# ------------------

ax3=fig.add_subplot(1,3,3)

numeric=\
df.select_dtypes(
include=np.number
)

corr=\
numeric.corr()

sns.heatmap(
corr,
cmap='coolwarm',
ax=ax3,
cbar=False
)

ax3.set_title(
"Correlation"
)

plt.tight_layout()

plt.savefig(
"dataset_overview.png",
dpi=150
)

plt.show()

print(
"\nĐã lưu dataset_overview.png"
)

print("\n"+"="*55)
print("HOÀN THÀNH")
print("="*55)