# Data Dictionary - Online Shoppers Dataset

## Chi tiết từng cột

| Tên cột | Kiểu dữ liệu | Mô tả | Giá trị hợp lệ | Ý nghĩa Business |
|---------|--------------|-------|----------------|------------------|
| Administrative | int64 | Số trang quản trị đã xem trong session | 0-27 | Đo mức độ quan tâm đến thông tin quản trị website |
| Administrative_Duration | float64 | Tổng thời gian trên các trang quản trị (giây) | 0-3398.75 | Thời gian dành cho việc tìm hiểu về công ty/website |
| Informational | int64 | Số trang thông tin đã xem | 0-24 | Số lần xem thông tin sản phẩm/dịch vụ |
| Informational_Duration | float64 | Tổng thời gian trên các trang thông tin (giây) | 0-2549.38 | Thời gian nghiên cứu thông tin trước khi mua |
| ProductRelated | int64 | Số trang sản phẩm đã xem | 0-705 | Số sản phẩm đã xem trong session |
| ProductRelated_Duration | float64 | Tổng thời gian trên các trang sản phẩm (giây) | 0-63973.52 | Thời gian xem xét sản phẩm |
| BounceRates | float64 | Tỷ lệ thoát (không tương tác) | 0-0.2 | Phần trăm session chỉ xem 1 trang rồi thoát |
| ExitRates | float64 | Tỷ lệ thoát từ trang | 0-0.2 | Trung bình tỷ lệ thoát từ các trang đã xem |
| PageValues | float64 | Giá trị trang (doanh thu tiềm năng) | 0-361.76 | Giá trị kinh tế của trang dựa trên e-commerce |
| SpecialDay | float64 | Độ gần với ngày đặc biệt | 0-1 | 1.0 = rất gần ngày lễ, 0 = không gần |
| Month | object | Tháng trong năm | Jan-Dec | Thời điểm trong năm |
| Month_encoded | float64 | Tháng mã hóa số | 1-12 | Dùng cho model ML |
| OperatingSystems | int64 | ID hệ điều hành | 1-8 | Loại OS của user |
| Browser | int64 | ID trình duyệt | 1-13 | Loại browser của user |
| Region | int64 | ID vùng địa lý | 1-9 | Vị trí địa lý của user |
| TrafficType | int64 | Loại traffic | 1-20 | Cách user đến website (organic, paid, direct, etc.) |
| VisitorType | object | Loại khách hàng | New_Visitor, Returning_Visitor, Other | Tần suất truy cập |
| VisitorType_encoded | int64 | Loại khách mã hóa | 0=New, 1=Other, 2=Returning | Dùng cho model ML |
| Weekend | int64 | Có phải cuối tuần | 0=No, 1=Yes | Thời điểm trong tuần |
| TotalTime | float64 | Tổng thời gian tất cả trang (giây) | 0-63973.52 | Tổng thời gian session |
| Revenue | int64 | Có mua hàng không | 0=No, 1=Yes | **TARGET VARIABLE** |

## Giải thích chi tiết các cột quan trọng

### BounceRates vs ExitRates
- **BounceRates**: Tỷ lệ session chỉ xem 1 trang rồi thoát
- **ExitRates**: Trung bình tỷ lệ thoát từ mỗi trang trong session
- **Ví dụ**: Session xem 3 trang, thoát ở trang cuối → ExitRates = 1/3 ≈ 0.33

### PageValues
- Giá trị kinh tế của trang
- Tính từ doanh thu thực tế của các session tương tự
- PageValues cao → trang quan trọng, khả năng mua cao

### SpecialDay
- 1.0: Rất gần ngày lễ (như Black Friday, Christmas)
- 0.8: Gần ngày lễ
- 0.6: Tương đối gần
- 0.4: Ít gần
- 0.2: Rất ít gần
- 0.0: Không gần ngày lễ nào

### TrafficType
- 1: Direct
- 2: Organic Search
- 3: Referral
- 4-20: Paid Search, Social Media, Email, etc.

### VisitorType
- **New_Visitor**: Khách hàng mới
- **Returning_Visitor**: Khách hàng quay lại
- **Other**: Khác (có thể là bot hoặc không xác định)

## Phân tích phân phối
- **Revenue**: 84.5% = 0 (Không mua), 15.5% = 1 (Có mua) → Imbalanced dataset
- **VisitorType**: ~85% Returning_Visitor, ~14% New_Visitor, ~1% Other
- **Month**: Thường có nhiều session vào tháng 5, 11 (tháng mua sắm)
- **Weekend**: ~23% session vào cuối tuần

## Lưu ý cho Modeling
- Dataset imbalanced → cần xử lý (SMOTE, undersampling, class weights)
- Các cột duration có outliers → có thể log transform
- Categorical columns đã được encode cho ML
- Missing values: Không có
- Duplicates: Đã loại bỏ