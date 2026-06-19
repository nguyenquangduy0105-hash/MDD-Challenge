# Phát Hiện và Chẩn Đoán Lỗi Phát Âm Tiếng Việt Dựa Trên Mạng Chú Ý Chéo Đa Luồng Dị Cấu

Dự án này được triển khai nhằm xử lý bài toán phát hiện và chuẩn đoán lỗi phát âm sai cho tiếng việt. Sử dụng cơ chế Cross attention giữa 4 luồng gồm:  
Chuỗi phoneme chuẩn và các thông tin đặc trưng âm học, cao độ và âm vị trích xuất thông qua wav2vec2 và mạng CNN, BiLSTM nông

## Cấu trúc thư mục
MDD Challenge  
&emsp;├── BuildVocab.py            
&emsp;├── EncodeCanonandTrans.py    
&emsp;├── G2P.py                   
&emsp;├── Model.py                 
&emsp;├── Trainer.py          
&emsp;└── Evaluate.py

## Cài đặt

1. Clone repository này:

```
git clone <url-repository-cua-ban>
cd <ten-thu-muc>
```
2. Cài đặt các thư viện:
```
pip install torch
pip install torchaudio
pip install numpy
pip install transformers
pip install pandas
pip install librosa
pip install pyarrow
pip install dataset
```
3. Chỉnh sửa các đường dẫn trong trong các file thành đường dẫn của bạn
> 🔴 **QUAN TRỌNG:** Vì mã nguồn được đóng gói từ môi trường Kaggle Notebook, trước khi chạy, bạn **bắt buộc** phải mở các file `.py` ra và chỉnh sửa lại cấu hình đường dẫn dữ liệu (`DATA_PATH`, `MODEL_PATH`,...) cho phù hợp với cấu trúc thư mục local trên máy của bạn.
## Chạy mô hình

Chạy pipeline theo các bước dưới đây để xử lý dữ liệu, huấn luyện và đánh giá mô hình:  
1. **Bước 1: Tiền xử lý dữ liệu**
    * Chạy `CreateMetadata.py` để tạo metadata từ dataset:  
      ```bash
      python CreateMetadata.py
      ```  
    * Chạy `G2P.py` để map từ word sang phoneme:  
      ```bash
      python G2P.py
      ```  
    * Chạy `BuildVocab.py` để tạo từ điển vocab:  
      ```bash
      python BuildVocab.py
      ```  
    * Chạy `EncodeCanonAndTrans.py` để encode canonical và transcript thành dãy số:  
      ```bash
      python EncodeCanonAndTrans.py
      ```  

2. **Bước 2: Huấn luyện mô hình**
    * Chạy `Trainer.py` để thực hiện huấn luyện mô hình:  
      ```bash
      python Trainer.py
      ```  

3. **Bước 3: Đánh giá mô hình**
    * Chạy `Evaluate.py` sau khi đã hoàn tất huấn luyện để đánh giá mô hình:  
      ```bash
      python Evaluate.py
      ```
