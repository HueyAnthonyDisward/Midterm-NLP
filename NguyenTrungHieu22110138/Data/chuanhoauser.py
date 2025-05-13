import pandas as pd
import os

# Đường dẫn file đầu vào và đầu ra
input_csv = "20191226-reviews.csv"  # Thay bằng đường dẫn thực tế
output_csv = "reviews_normalized.csv"

# Đọc file CSV
try:
    df = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"Không tìm thấy file {input_csv}. Vui lòng kiểm tra đường dẫn.")
    exit(1)

# Kiểm tra các cột cần thiết
required_columns = ['name', 'asin', 'rating']
if not all(col in df.columns for col in required_columns):
    print(f"File CSV thiếu một trong các cột: {required_columns}")
    exit(1)

# Báo cáo số hàng ban đầu
print(f"Số hàng ban đầu: {len(df)}")

# Kiểm tra giá trị null
print("\nKiểm tra giá trị null:")
print(df[required_columns].isnull().sum())
null_rows = df[df[required_columns].isnull().any(axis=1)]
if not null_rows.empty:
    print("\nCác hàng có giá trị null:")
    print(null_rows[required_columns])

# Loại bỏ các hàng có giá trị null trong các cột cần thiết
df = df.dropna(subset=required_columns)
print(f"Số hàng sau khi loại bỏ null: {len(df)}")

# Kiểm tra giá trị rating hợp lệ (1 đến 5)
invalid_ratings = df[~df['rating'].astype(float).between(1, 5)]
if not invalid_ratings.empty:
    print("\nCảnh báo: Các giá trị rating không hợp lệ (phải từ 1 đến 5):")
    print(invalid_ratings[['name', 'asin', 'rating']])
    df = df[df['rating'].astype(float).between(1, 5)]
    print(f"Số hàng sau khi loại bỏ rating không hợp lệ: {len(df)}")

# Kiểm tra trùng lặp dựa trên name và asin
duplicates = df[df.duplicated(subset=['name', 'asin'], keep=False)]
if not duplicates.empty:
    print("\nCảnh báo: Tìm thấy các hàng trùng lặp (name, asin):")
    print(duplicates[['name', 'asin', 'rating', 'date']])
    df = df.drop_duplicates(subset=['name', 'asin'], keep='last')
    print(f"Số hàng sau khi loại bỏ trùng lặp: {len(df)}")
else:
    print("\nKhông tìm thấy trùng lặp trong cặp name-asin.")

# Mã hóa userId và itemId thành số
user_mapping = {name: idx + 1 for idx, name in enumerate(df['name'].unique())}
item_mapping = {asin: idx + 1 for idx, asin in enumerate(df['asin'].unique())}

# Áp dụng mã hóa
df['userId'] = df['name'].map(user_mapping)
df['itemId'] = df['asin'].map(item_mapping)

# Chuyển rating thành kiểu số nguyên
df['rating'] = df['rating'].astype(int)

# Chỉ giữ các cột cần thiết
df_normalized = df[['userId', 'itemId', 'rating']]

# Kiểm tra lại trùng lặp trong userId-itemId
duplicates_normalized = df_normalized[df_normalized.duplicated(subset=['userId', 'itemId'], keep=False)]
if not duplicates_normalized.empty:
    print("\nLỗi: Vẫn còn trùng lặp trong userId-itemId sau khi mã hóa:")
    print(duplicates_normalized)
    exit(1)

# Báo cáo userId hợp lệ
print("\nDanh sách userId hợp lệ:")
print(df_normalized['userId'].unique())

# Kiểm tra dữ liệu sau khi chuẩn hóa
print("\nDữ liệu sau khi chuẩn hóa (5 dòng đầu):")
print(df_normalized.head())

# Lưu file CSV mới
df_normalized.to_csv(output_csv, index=False)
print(f"\nFile đã được chuẩn hóa và lưu tại: {output_csv}")

# Lưu ánh xạ userId và itemId
user_mapping_df = pd.DataFrame(list(user_mapping.items()), columns=['name', 'userId'])
item_mapping_df = pd.DataFrame(list(item_mapping.items()), columns=['asin', 'itemId'])
user_mapping_df.to_csv("user_mapping.csv", index=False)
item_mapping_df.to_csv("item_mapping.csv", index=False)
print("Đã lưu ánh xạ userId và itemId tại: user_mapping.csv, item_mapping.csv")