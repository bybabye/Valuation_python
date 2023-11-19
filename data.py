from faker import Faker
import pandas as pd
import random

fake = Faker()
Faker.seed(0)
random.seed(0)

# Danh sách các tiện ích có sẵn
available_utilities = ['bếp', 'điều hoà', 'nóng lạnh', 'tủ lạnh', 'máy giặt', 'tủ quần áo', 'wifi']
def generate_fake_address():
    return {
        'district': fake.city_suffix(),
        'ward': fake.street_name(),
        'city': 'Đà Nẵng',
    }
# Sinh dữ liệu giả lập
data = {
    'area': [random.randint(10, 40) for _ in range(10000)],
    'status': [random.choice([0, 1]) for _ in range(10000)],
    'roomType': ['tro' for _ in range(10000)],
    'utilities': [random.sample(available_utilities, k=random.randint(1, len(available_utilities))) for _ in range(10000)],
    'price': [random.randint(1000, 3000) // 100 * 100 for _ in range(10000)]  # Giữ giá có dạng XX00
}

# Tạo DataFrame
df = pd.DataFrame(data)
df.to_csv('data_can_ho.csv', index=False)
# Hiển thị dữ liệu
print(df.head())