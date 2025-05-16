import matplotlib.pyplot as plt

# Nhập số lượng thuật toán
n = int(input("Nhập số lượng thuật toán: "))

# Danh sách tên và thời gian
ten_thuat_toan = []
thoi_gian = []

# Nhập thông tin từng thuật toán
for i in range(n):
    ten = input(f"Nhập tên thuật toán thứ {i + 1}: ")
    tg = float(input(f"Nhập thời gian chạy của '{ten}' (đơn vị: giây): "))
    ten_thuat_toan.append(ten)
    thoi_gian.append(tg)

# Nhập tiêu đề biểu đồ
tieu_de = input("Nhập tiêu đề biểu đồ: ")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
bars = plt.bar(ten_thuat_toan, thoi_gian, color='skyblue')
plt.title(tieu_de)
plt.xlabel("Thuật toán")
plt.ylabel("Thời gian chạy (giây)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Thêm số liệu trên đầu cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}',
             ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()
