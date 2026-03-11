# README

## Giới thiệu

Dự án thử nghiệm nhận dạng chữ số (MNIST / Kaggle Digit Recognizer) với một số baseline models và nhóm model “nâng cao”.

## Yêu cầu hệ thống

- Python >= 3.8 (tại thời điểm build project này dùng 3.12)
- Thư viện Python:
  - `numpy`
  - `pandas`
  - `scikit-learn`
- Dataset:
  - `dataset/train.csv`
  - `dataset/test.csv`

## Cài đặt

1. Clone repository về máy:

    ```cmd
    git clone <đường dẫn repo>
    cd <tên folder dự án>
    ```

2. (Tùy chọn) Tạo và kích hoạt Python virtual environment:

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. Cài đặt các thư viện phụ thuộc:

   - Khuyến nghị (dùng `requirements.txt`):

     ```cmd
     pip install -r requirements.txt
     ```

   - Hoặc cài thủ công:

     ```cmd
     pip install numpy pandas scikit-learn matplotlib
     ```

## Sử dụng

- Chạy baseline models (nhẹ, nhanh):

  ```cmd
  python -m src.ml.run_models --mode base --train-csv dataset\train.csv --scale
  ```

- Chạy nhóm models “nâng cao” (nặng hơn, chậm hơn):

  ```cmd
  python -m src.ml.run_models --mode advanced --train-csv dataset\train.csv --scale
  ```

- Chạy model MLP (neural network, sklearn `MLPClassifier`):

  ```cmd
  python -m src.nn.run_mlp --train-csv dataset\train.csv --run-name mlp1 --hidden-layers 256,128 --max-iter 50
  ```

- Tuỳ chọn hữu ích:
  - `--test-size 0.2`: tỉ lệ validation split
  - `--seed 42`: random seed
  - `--scale`: scale pixel về 0..1 (thường giúp hội tụ nhanh hơn)
  - `--run-name demo1`: đặt tên run (tạo subfolder trong output)
  - `--outdir output\ml\base\demo1`: tự chỉ định thư mục output (ML)
  - `--outdir output\nn\mlp1`: tự chỉ định thư mục output (MLP/NN)

- Output:
  - Mặc định sẽ ghi vào `output/ml/<mode>/<timestamp>/`
  - File được tạo:
    - `scores.json`: gồm `meta` (tham số chạy) và `scores` (accuracy theo model)
    - `scores.txt`: bản text gọn để xem nhanh

- Output (MLP):
  - Mặc định sẽ ghi vào `output/nn/<timestamp>/` (hoặc theo `--run-name`)
  - Trong đó có `scores.json` và `scores.txt`

Ví dụ chạy và lưu theo tên run:

```cmd
python -m src.ml.run_models --mode base --scale --run-name demo1
```

## Đóng góp

Mọi đóng góp, phản hồi hoặc báo lỗi đều được hoan nghênh qua GitHub issues hoặc pull request.

## Giấy phép

Thông tin về giấy phép sử dụng mã nguồn (MIT, GPL, v.v.) hoặc để trống nếu chưa xác định.

## Liên hệ

Nếu cần hỗ trợ thêm, vui lòng liên hệ qua email hoặc thông tin Github của người phát triển.

*README này sẽ tiếp tục được cập nhật trong quá trình phát triển dự án.*
