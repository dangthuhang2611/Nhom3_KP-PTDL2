import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Đọc dữ liệu
df = pd.read_csv('stock_dataset.csv', header=1)
print(df)

df = df.drop(columns=['ID'])
df.info()

#Kiểm tra giá tr khuyết thiếu và trùng lặp
missing_values = df.isna().sum()
print(missing_values)
duplicated_values = df.duplicated().sum()
print(duplicated_values)

#Chuẩn hóa tên cột
df.columns = df.columns.str.strip().str.replace(' ', '_')
print("Các tên cột sau khi chuẩn hóa:")
print(df.columns)

#Chuẩn hóa định dạng dữ liệu
cols_to_convert = ['Annual_Return', 'Excess_Return', 'Total_Risk', 'Abs._Win_Rate', 'Rel._Win_Rate']
for col in cols_to_convert:
    #Xóa ký tự '%' và chuyển thành số thực
    df[col] = df[col].astype(str).str.replace('%', '', regex=False).astype(float) / 100

#Mô tả dữ liệu
print(df.describe())

#Xử lý giá trị ngoại lai

#Biểu đồ Box Plot của Annual Return
sns.boxplot(data=df['Annual_Return'],linewidth=1.5)
plt.title('Boxplot of Annual Return', fontsize=16)
plt.ylabel('Giá trị', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ Box Plot của Excess Return
sns.boxplot(data=df['Excess_Return'],linewidth=1.5)
plt.title('Boxplot of Excess_Return', fontsize=16)
plt.ylabel('Giá trị', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ Box Plot của Systematic Risk
sns.boxplot(data=df['Systematic_Risk'],linewidth=1.5)
plt.title('Boxplot of Systematic_Risk', fontsize=16)
plt.ylabel('Giá trị', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ Box Plot của Total Risk
sns.boxplot(data=df['Total_Risk'],linewidth=1.5)
plt.title('Boxplot of Total_Risk', fontsize=16)
plt.ylabel('Giá trị', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.show()
#%%
#Biểu đồ Box Plot của Abs. Win Rate
sns.boxplot(data=df['Abs._Win_Rate'],linewidth=1.5)
plt.title('Boxplot of Abs._Win_Rate', fontsize=16)
plt.ylabel('Giá trị', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ Box Plot của Rel. Win Rate
sns.boxplot(data=df['Rel._Win_Rate'],linewidth=1.5)
plt.title('Boxplot of Rel._Win_Rate', fontsize=16)
plt.ylabel('Giá trị', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

outlier_vars = ['Excess_Return', 'Systematic_Risk', 'Total_Risk']
for col in outlier_vars:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    print(f"Đã xử lý Capping cho biến: {col} | Ngưỡng dưới: {lower_bound:.4f} | Ngưỡng trên: {upper_bound:.4f}")

#Mô tả phân phối từng biến

#Biểu đồ histogram của Annual Return
plt.figure(figsize=(6, 4))
sns.histplot(df['Annual_Return'], kde=True, bins=20)
plt.title('Phân phối của Annual Return', fontsize=16)
plt.xlabel('Annual Return', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ histogram của Excess Return
plt.figure(figsize=(6, 4))
sns.histplot(df['Excess_Return'], kde=True, bins=20)
plt.title('Phân phối của Excess Return', fontsize=16)
plt.xlabel('Excess Return', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ histogram của Systematic Risk
plt.figure(figsize=(6, 4))
sns.histplot(df['Systematic_Risk'], kde=True, bins=20)
plt.title('Phân phối của Systematic_Risk', fontsize=16)
plt.xlabel('Systematic Risk', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ histogram của Total Risk
plt.figure(figsize=(6, 4))
sns.histplot(df['Total_Risk'], kde=True, bins=20)
plt.title('Phân phối của Total Risk', fontsize=16)
plt.xlabel('Total Risk', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ histogram của Abs. Win Rate
plt.figure(figsize=(6, 4))
sns.histplot(df['Abs._Win_Rate'], kde=True, bins=20)
plt.title('Phân phối của Abs. Win Rate', fontsize=16)
plt.xlabel('Abs. Win Rate', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Biểu đồ histogram của Rel. Win Rate
plt.figure(figsize=(6, 4))
sns.histplot(df['Rel._Win_Rate'], kde=True, bins=20)
plt.title('Phân phối của Rel. Win Rate', fontsize=16)
plt.xlabel('Rel. Win Rate', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Ma trận tương quan giữa các biến
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Ma trận tương quan giữa các biến', fontsize=16)
plt.show()

#Mối quan hệ giữa các biến
#Biểu đồ scatter plot giữa Abs. Win Rate và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Abs._Win_Rate', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Abs. Win Rate và Annual Return', fontsize=16)
plt.xlabel('Abs. Win Rate', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Excess Return và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Excess_Return', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Excess Return và Annual Return', fontsize=16)
plt.xlabel('Excess Return', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Rel. Win Rate và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Rel._Win_Rate', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Rel. Win Rate và Annual Return', fontsize=16)
plt.xlabel('Rel. Win Rate', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Large S/P và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Large_S/P', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Large S/P và Annual Return', fontsize=16)
plt.xlabel('Large S/P', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Large ROE và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Large_ROE', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Large ROE và Annual Return', fontsize=16)
plt.xlabel('Large ROE', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Large B/P và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Large_B/P', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Large B/P và Annual Return', fontsize=16)
plt.xlabel('Large B/P', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Systematic Risk và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Systematic_Risk', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Systematic Risk và Annual Return', fontsize=16)
plt.xlabel('Systematic Risk', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
plt.show()
#Biểu đồ scatter plot giữa Large Return Rate in the last quarter và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Large_Return_Rate_in_the_last_quarter', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Large Return Rate in the last quarter và Annual Return', fontsize=16)
plt.xlabel('Large Return Rate in the last quarter', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Small systematic Risk in the last quarter và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Small_systematic_Risk', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Small systematic Risk và Annual Return', fontsize=16)
plt.xlabel('Small systematic Risk', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Large Market Value và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Large_Market_Value', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Large Market Value và Annual Return', fontsize=16)
plt.xlabel('Large Market Value', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()
#Biểu đồ scatter plot giữa Total Risk và Annual Return
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Total_Risk', y='Annual_Return', data=df)
plt.title('Mối quan hệ giữa Total Risk và Annual Return', fontsize=16)
plt.xlabel('Total_Risk', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.show()

df.to_csv('cleaned_stock_dataset.csv', index=False)