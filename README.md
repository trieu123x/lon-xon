# CNN Lab — Assignment 05 (serverless)

**Đinh Hải Triều — B23DCCN843 — Lớp 06** · Intelligent System Development · TS. Trần Đình Quế

Ba mô hình CNN trên ba bộ dữ liệu Kaggle, gộp thành **một web tĩnh 3 trang**.
Suy luận chạy **100% trong trình duyệt** bằng [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
(WebAssembly). Không có API, không có máy chủ suy luận, không có bước build.

| Trang | Bộ dữ liệu (Kaggle) | Mô hình | Kết quả test |
|---|---|---|---|
| [`/cifar10`](cifar10.html) | [CIFAR-10](https://www.kaggle.com/datasets/quanbk/cifar10): 60.000 ảnh 32×32, 10 lớp | LeNet-5 · AlexNet · VGG-16 · **ResNet-50** | 69,1% · 86,0% · 91,5% · **92,2%** |
| [`/flowers`](flowers.html) | [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition): 4.307 ảnh, 5 loài | LeNet-5 · AlexNet · VGG-16 · **ResNet-50 (ImageNet)** | 66,2% · 65,7% · 83,6% · **93,0%** |
| [`/diabetes`](diabetes.html) | [Diabetes Prediction](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset): 100.000 hồ sơ | **CNN-1D** · MLP | ROC-AUC **0,9745** · 0,9749 |

Trang chủ (`/`) giới thiệu kiến trúc và có nút **tự kiểm chứng**: chạy mọi ảnh mẫu qua
mọi mô hình ngay trên trình duyệt, rồi so với xác suất mà Python/onnxruntime đã
ghi sẵn trong `models/*.json` (max |Δp| ≈ 1·10⁻⁶).

## Cấu trúc

```
index.html  cifar10.html  flowers.html  diabetes.html   4 trang tĩnh
app.js  style.css                                        mã dùng chung (không framework, không build)
models/*.onnx                                            10 mô hình, trọng số int8 theo kênh (≈ 89 MB)
models/{cifar10,flowers,diabetes}.json                   nhãn lớp, độ chính xác, ảnh mẫu + xác suất tham chiếu
samples/                                                 ảnh mẫu lấy từ tập test (PNG không nén mất mát)
vercel.json                                              cleanUrls + header cache cho .onnx
```

Mô hình được **commit thẳng vào repo** (file lớn nhất 23,7 MB, dưới giới hạn 100 MB
của GitHub), nên Vercel phục vụ chúng như mọi file tĩnh khác. Mỗi mô hình chỉ được
tải khi người dùng chọn đến nó.

## Deploy lên Vercel

1. [vercel.com/new](https://vercel.com/new) → **Import** repo này.
2. Framework Preset: **Other** · Build Command: *(để trống)* · Output Directory: *(để trống)*.
3. **Deploy.** Xong: `https://<tên>.vercel.app/`, `/cifar10`, `/flowers`, `/diabetes`.

## Chạy cục bộ

```bash
npx serve .          # hỗ trợ URL gọn /cifar10 giống Vercel
```

Không mở trực tiếp bằng `file://`: trình duyệt chặn `fetch()` các file mô hình.

## Mô hình được làm ra thế nào

Huấn luyện cục bộ bằng PyTorch 2.6 trên RTX 3060 Laptop (6 GB), AdamW + OneCycle +
AMP. Sau đó xuất ONNX (opset 17): BatchNorm được gộp vào Conv, lớp chuẩn hoá mean/std
nằm **trong** mô hình, nên web chỉ cần đưa ảnh [0, 1] vào. Trọng số được lượng tử hoá
int8 theo từng kênh đầu ra (`DequantizeLinear`), giúp file nhỏ ~4 lần trong khi
accuracy lệch ≤ 0,16 điểm. Notebook, báo cáo và phân tích từng biểu đồ nằm trong bài
nộp Assignment 05.
