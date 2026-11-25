# Hand Gesture Recognition – Điều khiển máy tính bằng cử chỉ tay

## 1. Giới thiệu đồ án

Đồ án xây dựng hệ thống nhận diện cử chỉ bàn tay thông qua camera, sau đó chuyển kết quả nhận diện thành các hành động điều khiển máy tính.

Mục tiêu là xây dựng mô hình có khả năng phân loại **11 cử chỉ tay đã được định nghĩa trước**.

Quy trình hoạt động tổng quát:

```text
Camera
   |
   v
Thu nhận hình ảnh bàn tay
   |
   v
Tiền xử lý hình ảnh
   |
   v
Mô hình nhận diện cử chỉ
   |
   v
Phân loại 11 gesture
   |
   v
Chuyển gesture thành hành động
   |
   v
Điều khiển máy tính
```

Đồ án tập trung vào nhận diện cử chỉ tĩnh thông qua hình ảnh. Các gesture được phân biệt bằng tư thế và hướng của bàn tay/ngón tay, không yêu cầu mô hình phải nhận diện chuỗi chuyển động theo thời gian.

## 2. Danh sách 11 gesture

Dataset hiện tại gồm 11 class (label):

| STT | Label | Ý nghĩa |
|---:|---|---|
| 1 | `MOUSE_UP` | Điều khiển chuột lên |
| 2 | `MOUSE_DOWN` | Điều khiển chuột xuống |
| 3 | `MOUSE_LEFT` | Điều khiển chuột sang trái |
| 4 | `MOUSE_RIGHT` | Điều khiển chuột sang phải |
| 5 | `SCROLL_UP` | Cuộn màn hình lên |
| 6 | `SCROLL_DOWN` | Cuộn màn hình xuống |
| 7 | `SCROLL_LEFT` | Cuộn màn hình sang trái |
| 8 | `SCROLL_RIGHT` | Cuộn màn hình sang phải |
| 9 | `CLICK` | Thực hiện thao tác click |
| 10 | `ZOOM_IN` | Phóng to |
| 11 | `ZOOM_OUT` | Thu nhỏ |

### Quy ước cử chỉ

- Nhóm `MOUSE`: sử dụng một ngón tay trỏ hướng theo bốn hướng.
- Nhóm `SCROLL`: sử dụng hai ngón tay hướng theo bốn hướng.
- `CLICK`: sử dụng tư thế OK.
- `ZOOM_IN`: sử dụng tư thế nắm bàn tay.
- `ZOOM_OUT`: sử dụng tư thế mở bàn tay.

**Lưu ý:** `NEUTRAL` không nằm trong 11 class của dataset hiện tại. Khi triển khai hệ thống, cần có cơ chế bỏ qua dự đoán không đủ tin cậy thay vì thực hiện hành động với mọi dự đoán. Confidence threshold cần được kiểm tra và hiệu chỉnh bằng dữ liệu phù hợp.

## 3. Dataset – Dữ liệu hình ảnh

### 3.1. Thành viên thu thập dữ liệu

Dataset hiện tại được tổ chức theo từng thành viên:

| Thành viên | Vai trò trong dataset |
|---|---|
| A | Thu thập dữ liệu cho Train và Validation |
| B | Thu thập dữ liệu cho Train và Validation |
| C | Thu thập dữ liệu cho Train và Validation |
| D | Thu thập dữ liệu riêng cho Test |

Số lượng ảnh giữa các thành viên có thể không bằng nhau. Ví dụ, một gesture có thể có 300 ảnh của A, 100 ảnh của B, 100 ảnh của C và 100 ảnh của D.

Các con số trên chỉ là ví dụ. Chương trình chia dataset sẽ tự đọc số ảnh thực tế trong từng thư mục, không yêu cầu mỗi thành viên phải có số lượng ảnh chính xác.

### 3.2. Cấu trúc dữ liệu gốc

Dữ liệu gốc được lưu trong `data/raw/`.

```text
data/
└── raw/
    ├── A/
    │   ├── CLICK/
    │   ├── MOUSE_UP/
    │   ├── MOUSE_DOWN/
    │   ├── MOUSE_LEFT/
    │   ├── MOUSE_RIGHT/
    │   ├── SCROLL_UP/
    │   ├── SCROLL_DOWN/
    │   ├── SCROLL_LEFT/
    │   ├── SCROLL_RIGHT/
    │   ├── ZOOM_IN/
    │   └── ZOOM_OUT/
    │
    ├── B/
    │   └── ... (11 gesture)
    │
    ├── C/
    │   └── ... (11 gesture)
    │
    └── D/
        └── ... (11 gesture)
```

Mỗi thư mục thành viên chứa 11 thư mục gesture. Hình ảnh được lưu trực tiếp trong thư mục label tương ứng.

Các định dạng ảnh được chương trình hỗ trợ:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### 3.3. Nguyên tắc chia Train / Validation / Test

Đồ án sử dụng cách chia dữ liệu theo thành viên để giữ tập Test độc lập với người tham gia huấn luyện.

**Train – 85% dữ liệu của A, B, C**

- Gộp ảnh của A, B, C theo từng gesture.
- Xáo trộn ảnh.
- Lấy 85% số ảnh của từng gesture cho Train.

**Validation – 15% dữ liệu của A, B, C**

- Sử dụng 15% ảnh còn lại của từng gesture.
- Dùng để theo dõi quá trình huấn luyện và đánh giá mô hình trong giai đoạn phát triển.

**Test – 100% dữ liệu của D**

- Toàn bộ ảnh của D được đưa vào Test.
- Không đưa ảnh của D vào Train hoặc Validation.
- Dùng để đánh giá khả năng nhận diện trên một người chưa xuất hiện trong Train và Validation.

### 3.4. Tỷ lệ chia dữ liệu

Tỷ lệ cấu hình trong chương trình:

```python
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
```

Thành viên D được dành riêng cho Test, tức là 100% ảnh của D thuộc Test.

Ví dụ, nếu một gesture có:

| Thành viên | Số ảnh |
|---|---:|
| A | 300 |
| B | 100 |
| C | 100 |
| D | 100 |
| Tổng | 600 |

Thì kết quả chia dự kiến:

| Dataset | Số ảnh |
|---|---:|
| Train | 425 |
| Validation | 75 |
| Test | 100 |
| Tổng | 600 |

Số lượng thực tế sẽ được tính riêng cho từng gesture dựa trên dữ liệu hiện có. Vì vậy, tổng ảnh của mỗi tập có thể không chia đều và không nhất thiết đạt đúng một tỷ lệ cố định trên toàn bộ dataset.

**Lưu ý về tỷ lệ:** Với ví dụ trên, tỷ lệ tổng thể là khoảng 70,8% Train, 12,5% Validation và 16,7% Test. Đây là kết quả có chủ đích vì ưu tiên Test độc lập theo người thay vì trộn dữ liệu của cùng một người vào cả ba tập.

## 4. Chương trình tự động chia dataset

File thực hiện chia dữ liệu:

```text
split_dataset.py
```

### 4.1. Chức năng

Chương trình thực hiện các công việc:

1. Đọc dữ liệu từ `data/raw/`.
2. Kiểm tra sự tồn tại của bốn thư mục thành viên A, B, C, D.
3. Kiểm tra các thành viên có cùng 11 thư mục gesture.
4. Đọc ảnh theo từng gesture.
5. Gộp ảnh của A, B, C.
6. Xáo trộn và chia 85% Train, 15% Validation.
7. Copy toàn bộ ảnh của D vào Test.
8. Thêm tiền tố thành viên vào tên ảnh để truy xuất nguồn gốc.
9. Thống kê số lượng ảnh theo từng gesture và từng tập dữ liệu.

### 4.2. Chạy chương trình

Mở Terminal tại thư mục gốc của project:

```bash
python split_dataset.py
```

Chương trình sử dụng `random seed = 42` để kết quả chia có thể tái lập khi chạy lại trên cùng dữ liệu đầu vào.

Nếu thư mục đầu ra đã chứa file, chương trình sẽ báo lỗi để tránh ghi đè hoặc nhân đôi dữ liệu. Muốn chia lại, cần kiểm tra và xóa thư mục `data/split/` cũ trước khi chạy.

### 4.3. Cấu trúc đầu ra

Sau khi chạy thành công:

```text
data/
├── raw/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── D/
│
└── split/
    ├── train/
    │   ├── CLICK/
    │   ├── MOUSE_UP/
    │   ├── MOUSE_DOWN/
    │   ├── MOUSE_LEFT/
    │   ├── MOUSE_RIGHT/
    │   ├── SCROLL_UP/
    │   ├── SCROLL_DOWN/
    │   ├── SCROLL_LEFT/
    │   ├── SCROLL_RIGHT/
    │   ├── ZOOM_IN/
    │   └── ZOOM_OUT/
    │
    ├── val/
    │   └── ... (11 gesture)
    │
    └── test/
        └── ... (11 gesture)
```

Mỗi tập có trực tiếp 11 thư mục gesture, không tạo thư mục A, B, C, D bên trong.

Ví dụ nội dung `train/CLICK/`:

```text
CLICK/
├── A__image_001.jpg
├── B__image_001.jpg
├── C__image_002.jpg
└── ...
```

Tên file có tiền tố thành viên để có thể nhận biết nguồn dữ liệu.

Ảnh được copy từ `raw` sang `split`, không xóa hoặc sửa đổi ảnh gốc.

## 5. Lưu ý về đánh giá mô hình

### 5.1. Data leakage

- Ảnh của D không xuất hiện trong Train hoặc Validation.
- Ảnh của A, B, C được chia ngẫu nhiên giữa Train và Validation.
- Vì vậy, Train và Validation có thể chứa ảnh của cùng một người.
- Test độc lập theo người D, phù hợp để kiểm tra khả năng tổng quát hóa sang người mới.

### 5.2. Giới hạn của phương pháp hiện tại

Hiện tại chỉ có một người (D) được dành riêng cho Test. Kết quả trên D chưa đại diện đầy đủ cho khả năng nhận diện trên tất cả người dùng.

Khi có thêm dữ liệu từ những người mới, có thể bổ sung họ vào Test mà không đưa ảnh của họ vào Train hoặc Validation.

Nếu ảnh được chụp liên tục trong cùng một phiên, các ảnh gần nhau có thể rất giống nhau. Khi có điều kiện, nên thu thập thêm ảnh ở các phiên khác nhau, góc tay khác nhau, khoảng cách camera khác nhau và điều kiện ánh sáng khác nhau.

### 5.3. Test phải được giữ độc lập

Không sử dụng Test để:

- Huấn luyện lại trọng số mô hình.
- Chọn hyperparameter.
- Điều chỉnh confidence threshold.
- Chọn mô hình dựa trên kết quả Test nhiều lần.

Các quyết định phát triển nên dựa trên Train và Validation. Test được sử dụng cho đánh giá cuối cùng.

## 6. Các bước tiếp theo của dự án

Các công việc tiếp theo dự kiến:

1. Kiểm tra chất lượng và số lượng dữ liệu của 11 gesture.
2. Chạy chương trình chia dataset và kiểm tra thống kê.
3. Tiền xử lý ảnh theo yêu cầu của mô hình.
4. Xây dựng và huấn luyện mô hình phân loại 11 gesture.
5. Theo dõi kết quả trên Validation.
6. Đánh giá cuối cùng trên Test.
7. Phân tích Confusion Matrix, Accuracy, Precision, Recall và F1-score.
8. Tích hợp mô hình vào chương trình nhận diện từ camera.
9. Xây dựng cơ chế confidence threshold để bỏ qua dự đoán không đáng tin cậy.
10. Kết nối kết quả nhận diện với các hành động điều khiển máy tính.

---

**Ghi chú:** README này mô tả trạng thái và quy trình chuẩn bị dữ liệu hiện tại. Kiến trúc mô hình, framework huấn luyện và cách triển khai API sẽ được cập nhật khi nhóm chốt các thành phần đó.


LƯU Ý ở dataset : Code đang lật ảnh webcam bằng cv2.flip(frame, 1) để thao tác giống gương.  thống nhất cách lật này trong suốt quá trình thu thập, không lúc lật lúc không.

