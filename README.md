# Neural Network from Scratch 

Mạng neural network cơ bản (Multi-Layer Perceptron) viết bằng C++ để huấn luyện và nhận dạng chữ số từ bộ dữ liệu MNIST, hoàn toàn không sử dụng framework bên ngoài.

---

## 1. Mục tiêu dự án

- **Đọc dữ liệu:** Giải mã định dạng file nhị phân chuẩn của MNIST (`idx3-ubyte` cho ảnh và `idx1-ubyte` cho nhãn).
- **Tiền xử lý:** Chuẩn hóa và co giãn kích thước ảnh đầu vào từ 40x40 về 28x28 pixel.
- **Khởi tạo tham số:** Tạo ngẫu nhiên ma trận trọng số (weights) và bias theo phân phối chuẩn.
- **Huấn luyện mạng:** Triển khai quá trình lan truyền xuôi (Forward), lan truyền ngược (Backpropagation) và tối ưu hóa bằng Mini-batch SGD.
- **Đánh giá & Lưu trữ:** Kiểm thử độ chính xác trên tập dữ liệu độc lập (Test set) và lưu tham số tối ưu ra file cấu hình.

---

## 2. Cấu trúc thư mục

```text
mainspace/
    └── main.cpp           # Luồng chạy chính (Vòng lặp huấn luyện, tính loss và xuất mô hình)

random_wb/
    └── exceed.cpp         # Khởi tạo trọng số và bias ngẫu nhiên

readmnist/
    └── read_mnist.cpp     # Đọc dữ liệu nhị phân và tiền xử lý ảnh
```
---

## 3. Kiến trúc mạng Neural
Lớp đầu vào (Input Layer): 784 nút (tương ứng với vector ảnh 28x28 pixel sau khi làm phẳng).

Lớp ẩn 1(Hidden Layer): 128 nút 

Lớp ẩn 2(Hidden Layer): 64 nút 

Lớp đầu ra (Output Layer): 10 nút (đại diện cho các chữ số từ 0 đến 9) k

---

## 4. Đánh giá hiệu năng
Độ chính xác trên tập kiểm tra (Test Accuracy): 92.0%

---

5. Hướng dẫn biên dịch và chạy
- Biên dịch dự án: ```g++ -std=c++11 mainspace/main.cpp random_wb/exceed.cpp readmnist/read_mnist.cpp -o mnist_nn```
- Chạy chương trình: ```./mnist_nn```
