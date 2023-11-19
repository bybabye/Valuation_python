import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('linear_regression_model.h5')

# Load the training data to preprocess the new data
train_data = pd.read_csv('data_can_ho_filter.csv')

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

# Thêm cột mới cho mỗi tiện ích trong dữ liệu huấn luyện
for col in utilities_mapping.values():
    train_data[col] = train_data['utilities'].apply(lambda x: 1 if col in x else 0)

# Tách dữ liệu thành đặc trưng (X) và biến phụ thuộc (Y)
train_X = train_data.drop(['utilities', 'price'], axis=1)

# Create a ColumnTransformer to apply One-Hot Encoding to the 'roomType' column
preprocessor = ColumnTransformer(
    transformers=[
        ('roomType', OneHotEncoder(), ['roomType'])
    ],
    remainder='passthrough'
)

# Apply the ColumnTransformer to transform the data
train_X_encoded = preprocessor.fit_transform(train_X)

# Standardize the data
scaler = StandardScaler()
scaler.fit(train_X_encoded)


def guess_price(new_data):
    

    # Thêm cột mới cho mỗi tiện ích trong dữ liệu mới
    for col in utilities_mapping.values():
        new_data[col] = new_data['utilities'].apply(lambda x: 1 if col in x else 0)

    # Tách dữ liệu mới thành đặc trưng (X) và biến phụ thuộc (Y)
    new_data = new_data.drop('utilities', axis=1)

    # Apply the ColumnTransformer to transform the new data
    new_data_encoded = preprocessor.transform(new_data)

    # Standardize the new data
    new_data_scaled = scaler.transform(new_data_encoded)

    # Predict house price for the new data
    predicted_price_new = model.predict(new_data_scaled)
    example_float = np.float32(predicted_price_new[0][0])
    float_as_python_type = float(example_float)
    return float_as_python_type

