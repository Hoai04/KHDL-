# Mô tả Dataset Online Shoppers Purchasing Intention

## Tổng quan Dataset
- **Nguồn**: UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset
- **Mục đích**: Phân tích hành vi người dùng trên website e-commerce để dự đoán khả năng mua hàng
- **Số dòng**: 12,205 (sau khi loại bỏ duplicates)
- **Số cột**: 21 (bao gồm 20 features + 1 target)

## Bài toán
- **Loại**: Phân loại nhị phân (Binary Classification)
- **Input**: Hành vi người dùng trong một session browsing
- **Output**: Khả năng mua hàng (Revenue: 0 = Không mua, 1 = Có mua)
- **Ứng dụng**: Tối ưu hóa trải nghiệm người dùng, cá nhân hóa khuyến mãi, cải thiện tỷ lệ chuyển đổi

## Các nhóm Feature chính

### 1. Nhóm hành vi (Behavior Features)
- **Administrative**: Số trang quản trị đã xem (int)
- **Administrative_Duration**: Tổng thời gian trên các trang quản trị (giây, float)
- **Informational**: Số trang thông tin đã xem (int)
- **Informational_Duration**: Tổng thời gian trên các trang thông tin (giây, float)
- **ProductRelated**: Số trang sản phẩm đã xem (int)
- **ProductRelated_Duration**: Tổng thời gian trên các trang sản phẩm (giây, float)

### 2. Nhóm chất lượng session (Session Quality)
- **BounceRates**: Tỷ lệ thoát trang (không tương tác) (float, 0-1)
- **ExitRates**: Tỷ lệ thoát từ trang (float, 0-1)
- **PageValues**: Giá trị trang (doanh thu tiềm năng từ trang) (float)

### 3. Nhóm thời gian (Temporal Features)
- **Month**: Tháng trong năm (string: Jan-Dec)
- **Month_encoded**: Tháng mã hóa (int: 1-12)
- **SpecialDay**: Độ gần với ngày đặc biệt (float, 0-1, càng gần càng cao)
- **Weekend**: Có phải cuối tuần không (int: 0 = Không, 1 = Có)

### 4. Nhóm kỹ thuật (Technical Features)
- **OperatingSystems**: ID hệ điều hành (int)
- **Browser**: ID trình duyệt (int)
- **TrafficType**: Loại traffic đến website (int)
- **Region**: Vùng địa lý (int)

### 5. Nhóm người dùng (User Features)
- **VisitorType**: Loại khách hàng (string: New_Visitor, Returning_Visitor, Other)
- **VisitorType_encoded**: Loại khách mã hóa (int: 0=New, 1=Other, 2=Returning)

### 6. Feature Engineering
- **TotalTime**: Tổng thời gian trên tất cả các loại trang (giây, float) = Administrative_Duration + Informational_Duration + ProductRelated_Duration

## Target Variable
- **Revenue**: Có tạo doanh thu không (int: 0 = FALSE, 1 = TRUE)

## Xử lý dữ liệu đã thực hiện
1. **Loại bỏ missing values**: Dataset gốc không có missing values
2. **Loại bỏ duplicates**: Đã loại bỏ các dòng trùng lặp
3. **Feature Engineering**:
   - Mã hóa Month thành Month_encoded
   - Mã hóa VisitorType thành VisitorType_encoded
   - Chuyển Weekend và Revenue thành int (0/1)
   - Tạo feature TotalTime
4. **Outlier Detection**: Sử dụng IQR method để phát hiện outliers (không loại bỏ, chỉ báo cáo)

## Insights quan trọng
- PageValues cao thường tương quan với Revenue = 1
- Returning visitors có tỷ lệ mua cao hơn
- Thời gian gần ngày đặc biệt (SpecialDay > 0) ảnh hưởng đến hành vi
- BounceRates và ExitRates thấp cho thấy session chất lượng cao

## File output
- **processed_online_shoppers.csv**: Dataset sạch sau xử lý
- **data_description.md**: File mô tả này
- **data_dictionary.md**: Chi tiết từng cột