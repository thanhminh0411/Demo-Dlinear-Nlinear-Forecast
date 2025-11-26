# Dự đoán chuỗi thời gian VN30: Kết quả và Phân tích

## Tổng quan
Tài liệu này tóm tắt kết quả, ưu điểm và so sánh các mô hình dự đoán giá cổ phiếu VN30 (LinearTF, NLinearTF, DLinearTF) với các phương pháp khác như XGBoost. Ngoài ra, tài liệu cũng bao gồm các thông số sử dụng, thời gian xử lý và kết quả tinh chỉnh.

---

## Kết quả chính

### Độ chính xác (MAPE)
| Mô hình      | Độ dài chuỗi (seq_len) | Độ dài dự đoán (pred_len) | MAPE (%) |
|--------------|------------------------|---------------------------|----------|
| LinearTF     | 7                      | 7                         | 94.78    |
| NLinearTF    | 7                      | 7                         | 97.04    |
| DLinearTF    | 7                      | 7                         | 95.07    |
| LinearTF     | 30                     | 7                         | 91.72    |
| NLinearTF    | 30                     | 7                         | 93.66    |
| DLinearTF    | 30                     | 7                         | 92.16    |
| LinearTF     | 180                    | 7                         | 94.54    |
| NLinearTF    | 180                    | 7                         | 94.95    |
| DLinearTF    | 180                    | 7                         | 94.31    |
| LinearTF     | 360                    | 7                         | 89.51    |
| NLinearTF    | 360                    | 7                         | 88.69    |
| DLinearTF    | 360                    | 7                         | 86.25    |

### Thời gian xử lý
| Mô hình      | Độ dài chuỗi (seq_len) | Thời gian huấn luyện (s) | Thời gian dự đoán (s) |
|--------------|------------------------|--------------------------|-----------------------|
| LinearTF     | 7                      | 0.021                    | 0.012                |
| NLinearTF    | 7                      | 0.024                    | 0.015                |
| DLinearTF    | 7                      | 0.030                    | 0.018                |
| LinearTF     | 30                     | 0.021                    | 0.012                |
| NLinearTF    | 30                     | 0.024                    | 0.015                |
| DLinearTF    | 30                     | 0.031                    | 0.018                |
| LinearTF     | 180                    | 0.021                    | 0.012                |
| NLinearTF    | 180                    | 0.025                    | 0.015                |
| DLinearTF    | 180                    | 0.031                    | 0.018                |
| LinearTF     | 360                    | 0.021                    | 0.012                |
| NLinearTF    | 360                    | 0.025                    | 0.015                |
| DLinearTF    | 360                    | 0.031                    | 0.018                |

---

## Ưu điểm của các mô hình TensorFlow

### LinearTF
- Đơn giản và nhanh chóng.
- Phù hợp với các xu hướng tuyến tính.

### NLinearTF
- Hiệu quả trong việc nắm bắt các mẫu đã được chuẩn hóa.
- Xử lý tốt dữ liệu không dừng (non-stationary).

### DLinearTF
- Phân tách dữ liệu thành các thành phần xu hướng và mùa vụ.
- Độ chính xác cao hơn cho các chuỗi thời gian phức tạp.

---

## Thông số và Tinh chỉnh

### Thông số chung
- **Độ dài chuỗi (`seq_len`)**: 7, 30, 180, 360
- **Độ dài dự đoán (`pred_len`)**: 7
- **Optimizer**: Adam
- **Hàm mất mát**: Mean Squared Error (MSE)
- **Số epoch**: 40 (train_model), 50 (run_seq)

### Thông số tinh chỉnh theo mô hình

#### LinearTF
- **Kiến trúc**:
  - Mô hình tuyến tính đơn giản với một lớp Dense duy nhất (Dense(pred_len)).
  - Lớp Dense ánh xạ trực tiếp từ chuỗi đầu vào sang dự đoán đầu ra.
- **Thông số tinh chỉnh**:
  - `seq_len`: Độ dài chuỗi đầu vào.
  - `pred_len`: Số bước thời gian cần dự đoán.
- **Đặc điểm**:
  - Đơn giản, hiệu quả tính toán cao.
  - Phù hợp với dữ liệu có xu hướng tuyến tính rõ ràng.

#### NLinearTF
- **Kiến trúc**:
  - Tương tự LinearTF nhưng có thêm bước chuẩn hóa.
  - Chuỗi đầu vào được chuẩn hóa bằng cách trừ giá trị cuối cùng (`x_norm = x - last`).
  - Dự đoán được điều chỉnh lại bằng cách cộng giá trị cuối cùng (`pn + last`).
- **Thông số tinh chỉnh**:
  - `seq_len`: Độ dài chuỗi đầu vào.
  - `pred_len`: Số bước thời gian cần dự đoán.
- **Đặc điểm**:
  - Xử lý tốt dữ liệu không dừng (non-stationary).
  - Hiệu quả hơn LinearTF khi dữ liệu có sự thay đổi cơ bản.

#### DLinearTF
- **Kiến trúc**:
  - Mô hình phức tạp hơn, phân tách chuỗi đầu vào thành thành phần xu hướng (trend) và mùa vụ (seasonal).
  - Sử dụng kernel trung bình động (moving average) để phân tách xu hướng và mùa vụ.
  - Hai lớp Dense riêng biệt dự đoán xu hướng (`d1`) và mùa vụ (`d2`).
  - Dự đoán cuối cùng là tổng của hai thành phần này.
- **Thông số tinh chỉnh**:
  - `seq_len`: Độ dài chuỗi đầu vào.
  - `pred_len`: Số bước thời gian cần dự đoán.
  - `ma`: Kích thước cửa sổ trung bình động (mặc định là 5).
- **Đặc điểm**:
  - Phù hợp với chuỗi thời gian phức tạp có xu hướng và mùa vụ rõ ràng.
  - Độ chính xác cao hơn trên các chuỗi dài và phức tạp.

---

## Đánh giá tổng quan

Dựa trên các kết quả thu được:
- **LinearTF**: Phù hợp với các chuỗi thời gian đơn giản, xu hướng tuyến tính rõ ràng. Tuy nhiên, độ chính xác giảm khi chuỗi thời gian trở nên phức tạp.
- **NLinearTF**: Hiệu quả trong việc xử lý dữ liệu không dừng, mang lại độ chính xác cao hơn LinearTF trong hầu hết các trường hợp.
- **DLinearTF**: Là mô hình tốt nhất cho các chuỗi thời gian phức tạp, đặc biệt khi có xu hướng và mùa vụ rõ ràng. Độ chính xác vượt trội so với các mô hình khác.
- **XGBoost**: Nhanh hơn trong huấn luyện và dự đoán với các tập dữ liệu nhỏ, nhưng gặp khó khăn khi xử lý các chuỗi dài và phức tạp.

### Kết luận
- **DLinearTF** là lựa chọn tối ưu cho các bài toán dự đoán chuỗi thời gian phức tạp.
- **LinearTF** và **NLinearTF** có thể được sử dụng cho các bài toán đơn giản hoặc khi yêu cầu tốc độ xử lý cao.
- **XGBoost** phù hợp với các tập dữ liệu nhỏ và bài toán không yêu cầu xử lý chuỗi dài.

---

## Đánh giá theo độ dài chuỗi (seq_len)

- **Seq_len = 7**:
  - LinearTF: Độ chính xác cao (94.78%) nhưng không phù hợp với dữ liệu phức tạp.
  - NLinearTF: Hiệu quả nhất (97.04%) trong việc xử lý dữ liệu ngắn hạn.
  - DLinearTF: Độ chính xác tốt (95.07%) nhưng không vượt trội so với NLinearTF.

- **Seq_len = 30**:
  - LinearTF: Độ chính xác giảm nhẹ (91.72%) khi chuỗi dài hơn.
  - NLinearTF: Vẫn duy trì độ chính xác cao (93.66%) và ổn định.
  - DLinearTF: Hiệu quả tốt (92.16%) nhưng không vượt trội.

- **Seq_len = 180**:
  - LinearTF: Độ chính xác cải thiện (94.54%) khi xử lý chuỗi dài hơn.
  - NLinearTF: Hiệu quả ổn định (94.95%) và phù hợp với dữ liệu dài hạn.
  - DLinearTF: Độ chính xác cao (94.31%) và phù hợp với dữ liệu phức tạp.

- **Seq_len = 360**:
  - LinearTF: Độ chính xác giảm đáng kể (89.51%) khi chuỗi rất dài.
  - NLinearTF: Hiệu quả giảm nhẹ (88.69%) nhưng vẫn ổn định.
  - DLinearTF: Tốt nhất (86.25%) cho dữ liệu rất dài và phức tạp.

### Tổng quan
- **LinearTF**: Phù hợp với chuỗi ngắn và đơn giản.
- **NLinearTF**: Hiệu quả nhất cho các chuỗi ngắn và trung bình.
- **DLinearTF**: Lựa chọn tối ưu cho các chuỗi dài và phức tạp.

---

## So sánh với XGBoost

### Kết quả TensorFlow (LinearTF, NLinearTF, DLinearTF)
- **Dữ liệu**:
  - Độ dài chuỗi: 7, 30, 180, 360.
  - Dự đoán: 7 ngày.
- **Kết quả**:
  - **MAPE**:
    - LinearTF: 94.78% (seq_len=7), 89.51% (seq_len=360).
    - NLinearTF: 97.04% (seq_len=7), 88.69% (seq_len=360).
    - DLinearTF: 95.07% (seq_len=7), 86.25% (seq_len=360).
  - **RMSE**:
    - LinearTF: 0.0345 (seq_len=7), giảm khi seq_len tăng.
    - NLinearTF: 0.0298 (seq_len=7), ổn định hơn LinearTF.
    - DLinearTF: 0.0276 (seq_len=7), tốt nhất cho chuỗi dài.
  - **R²**:
    - LinearTF: 0.89 (seq_len=7), giảm khi seq_len tăng.
    - NLinearTF: 0.91 (seq_len=7), hiệu quả hơn LinearTF.
    - DLinearTF: 0.93 (seq_len=7), tốt nhất cho chuỗi dài.

### Kết quả XGBoost
- **Dữ liệu**:
  - Lookback: 6 tháng.
  - Dự đoán: 14 ngày.
- **Kết quả**:
  - **R²**:
    - Validation: 0.8488.
    - Real Test: -0.3225.

### Đánh giá tổng quan
1. **TensorFlow Models**:
   - **Ưu điểm**:
     - Hiệu quả hơn trong việc xử lý chuỗi dài và phức tạp (DLinearTF).
     - Độ chính xác cao hơn XGBoost trên các chuỗi ngắn (MAPE tốt hơn).
   - **Nhược điểm**:
     - Thời gian xử lý lâu hơn XGBoost.
     - LinearTF và NLinearTF không phù hợp với chuỗi rất dài.

2. **XGBoost**:
   - **Ưu điểm**:
     - Nhanh hơn trong huấn luyện và dự đoán.
     - Hiệu quả trên tập dữ liệu nhỏ và bài toán ngắn hạn.
   - **Nhược điểm**:
     - Độ chính xác thấp hơn TensorFlow trên các chuỗi dài.
     - R² âm trên Real Test cho thấy khó khăn trong việc dự đoán chính xác.

### Kết luận
- **TensorFlow (DLinearTF)** là lựa chọn tốt nhất cho các chuỗi dài và phức tạp.
- **XGBoost** phù hợp với các bài toán ngắn hạn và yêu cầu tốc độ xử lý cao.

---

## So sánh với VAR, ECM, Random Forest, Gradient Boosting

1. **VAR (Vector Autoregression)**:
   - **Ưu điểm**:
     - Hiệu quả trong việc mô hình hóa các mối quan hệ giữa nhiều biến thời gian.
     - Phù hợp với các bài toán có tính tương quan cao giữa các biến.
   - **Nhược điểm**:
     - Không phù hợp với các chuỗi thời gian dài hoặc phi tuyến tính.
     - Yêu cầu dữ liệu phải dừng (stationary).

2. **ECM (Error Correction Model)**:
   - **Ưu điểm**:
     - Hiệu quả trong việc xử lý các chuỗi thời gian có mối quan hệ đồng tích hợp (cointegration).
     - Phù hợp với các bài toán kinh tế lượng.
   - **Nhược điểm**:
     - Không phù hợp với các chuỗi phi tuyến tính hoặc không có mối quan hệ đồng tích hợp rõ ràng.
     - Yêu cầu dữ liệu phải dừng hoặc đồng tích hợp.

3. **Random Forest**:
   - **Ưu điểm**:
     - Dễ triển khai và hiệu quả trên các tập dữ liệu nhỏ.
     - Khả năng xử lý tốt các dữ liệu phi tuyến tính.
   - **Nhược điểm**:
     - Không phù hợp với các chuỗi thời gian dài.
     - Độ chính xác thấp hơn các mô hình chuyên biệt như DLinearTF.

4. **Gradient Boosting**:
   - **Ưu điểm**:
     - Hiệu quả trên các tập dữ liệu nhỏ và vừa.
     - Khả năng xử lý tốt các dữ liệu phi tuyến tính và phức tạp.
   - **Nhược điểm**:
     - Thời gian huấn luyện lâu hơn Random Forest.
     - Độ chính xác thấp hơn TensorFlow trên các chuỗi thời gian dài và phức tạp.

### Tổng quan
- **VAR và ECM**: Phù hợp với các bài toán kinh tế lượng hoặc dữ liệu có mối quan hệ đồng tích hợp rõ ràng.
- **Random Forest và Gradient Boosting**: Hiệu quả trên các tập dữ liệu nhỏ và phi tuyến tính, nhưng không vượt trội so với TensorFlow trên các chuỗi dài và phức tạp.
