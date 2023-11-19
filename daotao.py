import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.callbacks import EarlyStopping
# Đọc dữ liệu từ file CSV
data = pd.read_csv('data_can_ho_filter.csv')


# Bảng ánh xạ từ số đến tên tiện ích
utilities_mapping = {
    1: 'điều hoà',
    2: 'nóng lạnh',
    3: 'tủ lạnh',
    4: 'máy giặt',
    5: 'tủ quần áo',
    6: 'bếp',
    7: 'wifi'
}
for col in utilities_mapping.values():
    data[col] = data['utilities'].apply(lambda x: 1 if col in x else 0)

# Tách dữ liệu thành đặc trưng (X) và biến phụ thuộc (Y)
X = data.drop(['price','utilities'], axis=1)
Y = data['price']

# Tạo ColumnTransformer để áp dụng One-Hot Encoding cho cột 'roomType'
preprocessor = ColumnTransformer(
    transformers=[
        ('roomType', OneHotEncoder(), ['roomType'])
    ],
    remainder='passthrough'
)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Áp dụng ColumnTransformer để chuyển đổi dữ liệu
X_encoded = preprocessor.fit_transform(X)



# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

# Tiêu chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng mô hình hồi quy tuyến tính
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1)
])
# Chọn optimizer và loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X_train_scaled, Y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, Y_test), callbacks=[early_stopping])

# Đánh giá mô hình trên tập kiểm tra
loss = model.evaluate(X_test_scaled, Y_test)
print(f'Loss trên tập kiểm tra: {loss}')

# Dự đoán giá trên một số mẫu từ tập kiểm tra
sample_indices = [0, 1, 2,3,4,5,6,7,8,9,10]
sample_data = X_test_scaled[sample_indices]
predicted_prices = model.predict(sample_data)

# Hiển thị kết quả
for i in range(len(sample_indices)):
    print(f"Dự đoán giá nhà: {predicted_prices[i][0]}, Giá thực tế: {Y_test.iloc[sample_indices[i]]}")


# Lưu mô hình vào tệp 'linear_regression_model.h5'
model.save('linear_regression_model.h5')
# Hiển thị thông tin về mô hình
model.summary()
