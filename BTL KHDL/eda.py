# ==============================
# 1. IMPORT
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 2. LOAD DATA
# ==============================
df_raw = pd.read_csv("online_shoppers.csv")
df = pd.read_csv("processed_online_shoppers.csv")

print("=== RAW DATA ===")
print(df_raw.head())

print("\n=== PROCESSED DATA ===")
print(df.head())

# ==============================
# 3. OVERVIEW
# ==============================
print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIBE ===")
print(df.describe())

# ==============================
# 4. EDA - PHÂN TÍCH CHÍNH
# ==============================

# 1. VisitorType vs Revenue
print("\n=== VisitorType vs Revenue ===")
print(df.groupby('VisitorType')['Revenue'].mean())

# 2. Weekend vs Revenue
print("\n=== Weekend vs Revenue ===")
print(df.groupby('Weekend')['Revenue'].mean())

# 3. Month vs Revenue
print("\n=== Month vs Revenue ===")
print(df.groupby('Month')['Revenue'].mean())

# 4. ExitRates vs Revenue
print("\n=== ExitRates vs Revenue ===")
print(df.groupby('Revenue')['ExitRates'].mean())

# 5. PageValues vs Revenue
print("\n=== PageValues vs Revenue ===")
print(df.groupby('Revenue')['PageValues'].mean())

# 6. TotalTime vs Revenue
print("\n=== TotalTime vs Revenue ===")
print(df.groupby('Revenue')['TotalTime'].mean())

# ==============================
# 5. VISUALIZATION
# ==============================

# VisitorType
plt.figure()
sns.barplot(x='VisitorType', y='Revenue', data=df)
plt.title("Revenue by Visitor Type")
plt.show()

# Month
plt.figure()
sns.barplot(x='Month', y='Revenue', data=df)
plt.title("Revenue by Month")
plt.xticks(rotation=45)
plt.show()

# Weekend
plt.figure()
sns.barplot(x='Weekend', y='Revenue', data=df)
plt.title("Revenue by Weekend")
plt.show()

# ExitRates
plt.figure()
sns.boxplot(x='Revenue', y='ExitRates', data=df)
plt.title("ExitRates vs Revenue")
plt.show()

# PageValues
plt.figure()
sns.boxplot(x='Revenue', y='PageValues', data=df)
plt.title("PageValues vs Revenue")
plt.show()

# TotalTime
plt.figure()
sns.boxplot(x='Revenue', y='TotalTime', data=df)
plt.title("Total Time vs Revenue")
plt.show()

# ==============================
# 6. CORRELATION
# ==============================
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ==============================
# 7. INSIGHT
# ==============================

print("\n=== INSIGHT ===")
print("""
New Visitor có tỷ lệ mua cao nhất (~24.9%), cao hơn Returning Visitor (~14%)
→ Điều này cho thấy khách hàng mới có xu hướng mua ngay trong lần truy cập đầu tiên
Người truy cập vào cuối tuần có tỷ lệ mua cao hơn
→ Weekend (17.4%) > Weekday (15.0%)
Tháng 11 (Nov) có tỷ lệ mua cao nhất (~25.5%)
→ Có thể do các chương trình khuyến mãi lớn (Black Friday, sale cuối năm)
ExitRates thấp hơn ở nhóm mua hàng
→ Người mua thường không rời trang sớm
PageValues cao ở nhóm mua (~27 vs ~2)
→ Đây là chỉ số dự đoán mạnh hành vi mua
TotalTime cao → khả năng mua cao
→ Người dùng dành nhiều thời gian hơn có xu hướng chuyển đổi
""")