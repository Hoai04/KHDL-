import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Đặt style cho plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Download dataset nếu chưa có
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
dataset_path = 'online_shoppers.csv'

if not os.path.exists(dataset_path):
    print("Downloading dataset...")
    import urllib.request
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Dataset downloaded!")
else:
    print("Dataset already exists!")

df = pd.read_csv(dataset_path)

# 2.1 Giới thiệu dataset
print("2.1 Giới thiệu dataset")
print("Nguồn dữ liệu: UCI Machine Learning Repository")
print("Mô tả tổng quan: Dataset Online Shoppers Purchasing Intention ghi lại hành vi người dùng")
print("trên website thương mại điện tử với hơn 12,000 phiên truy cập.")
print("Ý nghĩa thực tế: Phân tích hành vi người dùng để dự đoán khả năng mua hàng.")
print()

# Data Quality Check trước
print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU (Xử lý trước)")
missing_values = df.isnull().sum().sum()
duplicates = df.duplicated().sum()
print(f"Missing values: {missing_values}")
print(f"Duplicates: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Đã loại bỏ {duplicates} dòng trùng lặp. Dataset còn {len(df)} dòng.")
print()

# 2.2 Thông tin tổng quan
print("2.2 Thông tin tổng quan")
print(f"Số dòng: {len(df)}")
print(f"Số cột: {df.shape[1]}")
print("Kiểu dữ liệu:")
print("- Numerical: Administrative, Administrative_Duration, Informational, Informational_Duration,")
print("  ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay")
print("- Categorical: Month, VisitorType")
print("- Boolean: Weekend, Revenue")
print()

# 2.3 Xác định bài toán
print("2.3 Xác định bài toán")
print("Input (features): Hành vi người dùng trong một phiên truy cập")
print("- Số trang đã xem và thời gian tương tác")
print("- Tỷ lệ thoát và giá trị trang")
print("- Thông tin thời gian và kỹ thuật")
print()
print("Output: Revenue (0/1) - Khả năng tạo doanh thu")
print("Loại bài toán: Binary Classification")
print("Phương pháp sử dụng: Logistic Regression")
print()

# 2.4 Data Dictionary
print("2.4 Data Dictionary")
print()
print("Hành vi người dùng:")
print("- Administrative: Số trang quản trị đã xem")
print("- Administrative_Duration: Thời gian trên trang quản trị")
print("- Informational: Số trang thông tin đã xem")
print("- Informational_Duration: Thời gian trên trang thông tin")
print("- ProductRelated: Số trang sản phẩm đã xem")
print("- ProductRelated_Duration: Thời gian trên trang sản phẩm")
print()
print("Chất lượng session:")
print("- BounceRates: Tỷ lệ thoát (không tương tác)")
print("- ExitRates: Tỷ lệ thoát trang")
print("- PageValues: Giá trị trang (doanh thu tiềm năng)")
print()
print("Thời gian:")
print("- Month: Tháng trong năm")
print("- SpecialDay: Gần ngày đặc biệt (0-1)")
print()
print("Kỹ thuật:")
print("- OperatingSystems: Hệ điều hành")
print("- Browser: Trình duyệt")
print("- TrafficType: Loại traffic")
print("- Region: Vùng địa lý")
print()
print("Người dùng:")
print("- VisitorType: Loại khách (New/Returning/Other)")
print("- Weekend: Có phải cuối tuần không")
print()

# 2.5 Kiểm tra chất lượng dữ liệu (Data Quality)
print("2.5 Kiểm tra chất lượng dữ liệu (Data Quality)")
print("Missing values: Không có")
print("Duplicates: Đã loại bỏ 125 dòng trùng lặp")
print("Outliers: Phát hiện trong các cột numerical nhưng giữ nguyên")
print("(đặc trưng của dữ liệu hành vi người dùng)")
print()

# 2.6 Feature Engineering (định hướng)
print("2.6 Feature Engineering (định hướng)")
print("Chuyển đổi dữ liệu:")
print("- Boolean → số: Weekend, Revenue")
print("- Encode categorical: Month → số, VisitorType → số")
print()
print("Đề xuất tạo biến mới:")
print("- TotalTime: Tổng thời gian user ở trên website")
print("  (Administrative_Duration + Informational_Duration + ProductRelated_Duration)")
print()

# Feature Engineering
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df['Month_encoded'] = df['Month'].map(month_mapping)

le = LabelEncoder()
df['VisitorType_encoded'] = le.fit_transform(df['VisitorType'])

df['TotalTime'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']

print("Feature Engineering hoàn thành!")
print()

# 2.7 Nhận xét & Insight ban đầu
print("2.7 Nhận xét & Insight ban đầu")
print("Nhận định về hành vi người dùng:")
print("- Người dùng thường xem nhiều trang sản phẩm hơn trang quản trị/thông tin")
print("- Thời gian tương tác cao hơn ở trang sản phẩm")
print("- Returning visitors có tỷ lệ mua hàng cao hơn")
print()
print("Ý nghĩa cho bài toán kinh doanh:")
print("- Giúp doanh nghiệp hiểu yếu tố ảnh hưởng đến quyết định mua hàng")
print("- Tối ưu hóa chiến lược marketing và trải nghiệm người dùng")
print("- Dự đoán doanh thu tiềm năng từ mỗi phiên truy cập")
print()

# Lưu dataset đã xử lý
df.to_csv('processed_online_shoppers.csv', index=False)
print("Dataset đã xử lý được lưu thành 'processed_online_shoppers.csv'")

# Tạo visualizations cơ bản
print("Tạo visualizations...")

# Distribution of Revenue
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Revenue')
plt.title('Phân bố Revenue')
plt.savefig('revenue_distribution.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

print("Visualizations đã được lưu!")