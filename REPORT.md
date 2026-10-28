# Phân loại trạng thái mắt bằng CNN

## 1. Mục tiêu

Xây dựng mô hình học sâu phân loại ảnh mắt thành hai lớp `awake` (tỉnh táo) và `sleepy` (buồn ngủ). Đề tài tập trung vào phân loại ảnh mắt tĩnh; phiên bản hiện tại chưa nhận diện khuôn mặt, chưa xử lý video/webcam và chưa phải hệ thống cảnh báo buồn ngủ hoàn chỉnh.

## 2. Dữ liệu

Dataset MRL Eye Images có sẵn được giữ nguyên trên đĩa. Nhãn lấy từ thư mục `awake/sleepy`; khi huấn luyện, chương trình gom các đường dẫn trong bộ nhớ và chia theo subject ID trong filename, không di chuyển hay ghi đè ảnh.

| Tập | Awake | Sleepy | Tổng |
|---|---:|---:|---:|
| Train | 25.770 | 25.167 | 50.937 |
| Validation | 8.591 | 8.389 | 16.980 |
| Test | 8.591 | 8.390 | 16.981 |
| **Tổng** | **42.952** | **41.946** | **84.898** |

Ảnh được chuyển sang grayscale, đổi kích thước 80 x 80 và chuẩn hóa giá trị pixel về [0, 1]. Khi huấn luyện, ảnh có thể được lật ngang ngẫu nhiên.

## 3. Phương pháp

Mô hình được triển khai bằng PyTorch. CNN gồm ba khối tích chập 3 x 3 với lần lượt 32, 64 và 128 kênh; mỗi khối sử dụng Batch Normalization, ReLU và Max Pooling. Adaptive Average Pooling đưa đặc trưng về kích thước 1 x 1, sau đó Dropout 0,3 và lớp tuyến tính dự đoán hai lớp.

Mô hình dùng Cross Entropy và AdamW với learning rate 0,001, batch size 128 và seed 42. Lần đánh giá cuối dùng tối đa 5 epoch để phù hợp thời gian chạy CPU. Subject ID được chia rời nhau giữa train/validation/test (tỷ lệ xấp xỉ 70/15/15), đồng thời giữ cả hai lớp trong mỗi tập. Checkpoint tốt nhất theo validation accuracy được lưu tại `models/eye_state_cnn.pt`; tập test chỉ được đánh giá sau khi huấn luyện xong.

Sau khi chọn checkpoint, một nhiệt độ duy nhất được fit bằng logits trên validation set để hiệu chuẩn softmax confidence. Cùng nhiệt độ đó được lưu trong checkpoint và dùng cho phần trăm ở giao diện. Tập test không được dùng để chọn checkpoint hay fit nhiệt độ.

| Tập dùng trong thí nghiệm | Subject | Awake | Sleepy | Tổng |
|---|---:|---:|---:|---:|
| Train | 26 | 30.286 | 30.083 | 60.369 |
| Validation | 6 | 6.209 | 5.997 | 12.206 |
| Test | 5 | 6.457 | 5.866 | 12.323 |

Các subject giữa ba tập không giao nhau. Thư mục ảnh gốc không bị thay đổi; việc nhóm diễn ra trong bộ nhớ theo tiền tố subject trong tên file.

## 4. Kết quả

Thiết bị: CPU (PyTorch 2.14.0+cpu). Huấn luyện 5 epoch; checkpoint tốt nhất ở epoch 5, validation accuracy 89,72%. Temperature scaling được fit riêng trên validation, với $T = 1,0342$; validation NLL giảm từ 0,2385 xuống 0,2384. Test gồm 12.323 ảnh từ 5 subject không có trong train/validation. Sau hiệu chuẩn, test accuracy là 89,91%, test loss là 0,2730.

| Chỉ số | Kết quả |
|---|---:|
| Validation accuracy tốt nhất | 89,72% (epoch 5) |
| Temperature (fit trên validation) | 1,0342 |
| Validation NLL trước / sau hiệu chuẩn | 0,2385 / 0,2384 |
| Test loss (sau hiệu chuẩn) | 0,2730 |
| Test accuracy | 89,91% |
| Awake: precision / recall / F1 | 95,80% / 84,45% / 89,77% |
| Sleepy: precision / recall / F1 | 84,86% / 95,93% / 90,05% |

Confusion matrix (hàng là nhãn thật, cột là nhãn dự đoán):

| Nhãn thật \\ Dự đoán | Awake | Sleepy |
|---|---:|---:|
| Awake | 5.453 | 1.004 |
| Sleepy | 239 | 5.627 |

## 5. Demo ảnh

Ứng dụng dùng Haar Cascade của OpenCV để dò khuôn mặt lớn nhất trong ảnh. Từ khung khuôn mặt, chương trình ước lượng hai vùng mắt trái/phải, đưa từng vùng vào CNN và vẽ ellipse màu trực tiếp lên ảnh: xanh cho `awake`, đỏ cho `sleepy`. Mỗi mắt hiển thị nhãn và softmax confidence đã hiệu chuẩn trên validation. Trong ảnh test hồng ngoại, recall của lớp `sleepy` là 95,93%; với selfie màu, confidence không đảm bảo xác suất đúng vì khác miền dữ liệu. Nếu không tìm thấy khuôn mặt, ứng dụng yêu cầu chọn ảnh rõ và chính diện thay vì phân loại nhầm toàn ảnh thành ảnh mắt.

## 6. Hạn chế và hướng phát triển

- Số subject chỉ là 37; test theo subject độc lập phù hợp hơn nhưng vẫn có độ bất định do số subject test nhỏ.
- CNN được huấn luyện trên ảnh mắt hồng ngoại cắt sẵn, còn ảnh demo có thể là ảnh màu từ điện thoại; khác biệt miền dữ liệu khiến dự đoán ngoài tập test kém tin cậy hơn.
- Vùng mắt được ước lượng từ tỷ lệ khung mặt, chưa phải mô hình landmark chuyên dụng. Góc nghiêng, che khuất hoặc ánh sáng xấu có thể làm vùng khoanh lệch.
- Ứng dụng hiện xử lý ảnh tĩnh. Hướng phát triển là huấn luyện thêm bằng ảnh webcam/điện thoại và xử lý video theo thời gian.
- Kết quả phân loại không nên được dùng như hệ thống an toàn duy nhất để quyết định người lái xe có tỉnh táo hay không.

## 7. Chạy chương trình

Từ thư mục gốc dự án:

```powershell
& 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe' 'data\train_model.py' --epochs 10 --batch-size 64
```

Phân loại một ảnh bằng checkpoint đã lưu:

```powershell
& 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe' 'data\train_model.py' --predict 'duong-dan\anh-mat.png'
```