import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


df = pd.read_csv("cleaned_stock_dataset.csv")

#Chia tập huấn luyện và tập kiểm tra
y = df['Annual_Return']
X = df.drop(columns=['Annual_Return', 'Excess_Return', 'Systematic_Risk'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Size of X_train: ", X_train.shape)
print("Size of X_test: ", X_test.shape)
print("Size of y_train: ", y_train.shape)
print("Size of y_test: ", y_test.shape)

#Chuẩn bị mô hình thực nghiệm
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

model = TransformedTargetRegressor(
    regressor=pipeline,
    func=None,
    inverse_func=None
)

#Khởi tạo K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Thực hiện kiểm định chéo
cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
cv_neg_rmse = cross_val_score(model, X_train, y_train, cv=kf,
                              scoring='neg_root_mean_squared_error')
cv_neg_mae = cross_val_score(model, X_train, y_train, cv=kf,
                             scoring='neg_mean_absolute_error')
cv_rmse = -cv_neg_rmse
cv_mae = -cv_neg_mae

#Kết quả kiểm định chéo
results = {
    "R2": cv_r2,
    "RMSE": cv_rmse,
    "MAE": cv_mae
}

df_results = pd.DataFrame(results)
df_results.loc["Average"] = [
    cv_r2.mean(), cv_rmse.mean(), cv_mae.mean()
]
print("Cross-validation results:")
print(df_results)

#Huấn luyện mô hình chính
model.fit(X_train, y_train)
print("Hệ số hồi quy của mô hình: ")
print(model.regressor_['model'].coef_)
print("Hệ số chặn của mô hình: ")
print(model.regressor_['model'].intercept_)

#Đánh giá trên tập kiểm tra
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Đánh giá trên tập kiểm tra")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

#Lưu mô hình
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Mô hình đã được huấn luyện và lưu thành công!")

#Vẽ biểu đồ scatter so sánh giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.title('Giá trị thực tế vs Giá trị dự đoán')
plt.xlabel('Annual Return thực tế')
plt.ylabel('Annual Return dự đoán')
plt.grid(True)
plt.show()

#Vẽ biểu đồ QQ-plot
all_residuals = []
all_fitted = []
residuals = y_test - y_pred
all_residuals.extend(residuals)
all_fitted.extend(y_pred)
stats.probplot(all_residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals (K-fold CV)")
plt.show()

#Vẽ biểu đồ phần dư
plt.figure(figsize=(8, 6))
sns.scatterplot(x=all_fitted, y=all_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Phần dư')
plt.grid(True)
plt.show()
