
import pandas as pd

def filter_data(input_csv, output_csv):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(input_csv)

    # Điều kiện loại bỏ
    # area > 25  , nội thất đầy đủ giá không thể dướI 2 triệu
    # area < 15 , giá không thể trên 2 triệu 5 cho dù nội thất đầy đỦ
    # area > 25 , giá không thể dưới 1 triệu 5
    condition_to_remove = (
        ((data['area'] > 25) & (data['utilities'].apply(eval).apply(len) > 4) & (data['price'] < 2000)) |
        ((data['area'] < 15) & (data['price'] > 2500)) |
        ((data['area'] > 25) & (data['price'] < 1500)) |
        ((data['area'] > 30) & (data['price'] < 1800)) |
        ((data['area'] > 30) & (data['utilities'].apply(eval).apply(len) > 5 & (data['price'] < 2500)))
    )

   

     # Xóa các dòng trùng lặp
    filtered_data_trunglap = data.drop_duplicates()
    # Lọc dữ liệu
    filtered_data = filtered_data_trunglap[~condition_to_remove]
    # Lưu dữ liệu đã lọc vào file CSV mới
    filtered_data.to_csv(output_csv, index=False)

# Gọi hàm để loại bỏ dữ liệu
filter_data('data_can_ho.csv', 'data_can_ho_filter.csv')
