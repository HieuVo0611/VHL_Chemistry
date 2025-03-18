# VHL_Chemistry

## Giới thiệu

VHL_Chemistry là một dự án AI được thiết kế để phân tích dữ liệu hóa học, bao gồm các mô hình dự đoán và xử lý dữ liệu từ các tệp CSV và Excel. Dự án bao gồm các công cụ cho việc xử lý dữ liệu, huấn luyện mô hình, tối ưu hóa siêu tham số, và dự đoán trên các mẫu mới.

## Tính năng chính

1. **Xử lý dữ liệu Pb (Lead):**

   - Đọc dữ liệu từ các tệp `.swp` và chuyển đổi thành định dạng CSV.
   - Tạo các biểu đồ minh họa dữ liệu.
   - Xây dựng tập dữ liệu với kỹ thuật padding và trích xuất nhãn từ tên tệp.
2. **Mô hình dự đoán Pb:**

   - Chuẩn bị dữ liệu.
   - Huấn luyện các mô hình như Linear Regression, Random Forest, Gradient Boosting và XGBoost.
   - Tối ưu hóa siêu tham số bằng GridSearchCV và RandomizedSearchCV.
   - Đánh giá mô hình và dự đoán trên tập kiểm tra và mẫu mới.
3. **Xử lý và dự đoán dữ liệu ENR-CIP:**

   - Đọc dữ liệu từ các tệp Excel.
   - Chuẩn hóa dữ liệu và xử lý padding.
   - Huấn luyện mô hình XGBoost để dự đoán nồng độ ENR và CIP từ dữ liệu.

## Cấu trúc dự án

VHL_Chemistry/
├── src/
│	├── pb/
│	│	├── pb_model.ipynb 			# Huấn luyện và dự đoán dữ liệu Pb
│	│	├── pb_processing_data.ipynb 	# Xử lý dữ liệu Pb từ tệp .swp
│	├── cip_enr/
│		├── cip_enr_model.py			# Huấn luyện và dự đoán dữ liệu ENR-CIP
├── data/
│	├── pb/ 							# Dữ liệu Pb
│	├── cip_enr/ 						# Dữ liệu ENR-CIP
├── README.md	 					# Tài liệu dự án
├── LICENSE 						# Giấy phép sử dụng
├── requirements.txt 					# Danh sách các thư viện cần thiết

## Cài đặt

1. **Clone dự án:**

   ```bash
   git clone git@github.com:HieuVo0611/VHL_Chemistry.git
   cd VHL_Chemistry
   ```
2. **Cài đặt các thư viện: Sử dụng pip để cài đặt các thư viện từ requirements.txt:**

   ```python-repl
   pip install -r requirements.txt
   ```
3. **Cấu trúc thư mục dữ liệu:**

* Đặt dữ liệu Pb trong thư mục `data/pb/`.
* Đặt dữ liệu ENR-CIP trong thư mục `data/cip_enr/`.

## Sử dụng

* Xử lý dữ liệu Pb: Chạy notebook `src/pb/pb_processing_data.ipynb` để chuyển đổi và xử lý dữ liệu Pb từ các tệp `.swp`.
* Huấn luyện và dự đoán Pb: Chạy notebook `src/pb/pb_model.ipynb` để huấn luyện mô hình và dự đoán.
* Xử lý và huấn luyện ENR-CIP: Chạy script `src/cip_enr/cip_enr_model.py` để xử lý dữ liệu và huấn luyện mô hình dự đoán ENR-CIP.

## Yêu cầu hệ thống

* Python 3.9 trở lên.
* Các thư viện Python được liệt kê trong `requirements.txt`.

## Giấy phép

* Dự án được cấp phép theo MIT License.
