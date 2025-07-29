import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import ttest_rel

# Đường dẫn đến file dữ liệu
file_path = 'D:/Khóa luận/data/filtered_data.csv'

# Đọc dữ liệu từ file CSV
df = pd.read_csv(file_path)

# Danh sách cột hóa học
chemical_columns = [
    'Cu', 'Al', 'Ag', 'B', 'Be', 'Ca', 'Co', 'Ce', 'Cr', 'Fe', 'Hf', 'La', 'Mg', 
    'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pr', 'Si', 'Sn', 'Ti', 'V', 'Zn', 'Zr'
]

# Các cột điều kiện và độ dẫn điện
condition_columns = ['Solid_Solution_Temp_K', 'Aging_Temp_K', 'Aging_Time_h']
conductivity_column = 'Electrical_Conductivity_IACS'

# Kết hợp tất cả các cột đặc trưng
feature_columns = chemical_columns + condition_columns

# Kiểm tra xem các cột có tồn tại không
missing_columns = [col for col in feature_columns + [conductivity_column] if col not in df.columns]
if missing_columns:
    print(f"Không tìm thấy các cột: {', '.join(missing_columns)}")
    exit()

# Chọn dữ liệu đặc trưng và mục tiêu
X = df[feature_columns]
y = df[conductivity_column]

# Kiểm tra giá trị NaN
if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
    print("Dữ liệu chứa giá trị NaN. Vui lòng kiểm tra và làm sạch dữ liệu.")
    exit()

# Danh sách các random_state để thử nghiệm
random_states = [42, 123, 456]

# Lưu kết quả
results = {
    'random_state': [],
    'R2_train': [],
    'MSE_train': [],
    'RMSE_train': [],
    'R2_test': [],
    'MSE_test': [],
    'RMSE_test': [],
    'R2_cv': [],
    'MSE_cv': [],
    'RMSE_cv': []
}

# Chạy mô hình với từng random_state
for rs in random_states:
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    
    # Khởi tạo và huấn luyện mô hình
    rf_model = RandomForestRegressor(n_estimators=100, random_state=rs)
    rf_model.fit(X_train, y_train)
    
    # Dự đoán trên tập huấn luyện và kiểm tra
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Tính R² và MSE
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    cv_scores_mse = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
    r2_cv = cv_scores.mean()
    mse_cv = -cv_scores_mse.mean()
    rmse_cv = np.sqrt(mse_cv)
    
    # Lưu kết quả
    results['random_state'].append(rs)
    results['R2_train'].append(r2_train)
    results['MSE_train'].append(mse_train)
    results['RMSE_train'].append(rmse_train)
    results['R2_test'].append(r2_test)
    results['MSE_test'].append(mse_test)
    results['RMSE_test'].append(rmse_test)
    results['R2_cv'].append(r2_cv)
    results['MSE_cv'].append(mse_cv)
    results['RMSE_cv'].append(rmse_cv)

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)

# In kết quả
print("\nKết quả cho các random_state khác nhau:")
print(results_df)

# Paired t-test để so sánh R² và MSE giữa các lần chạy
# So sánh R² trên tập kiểm tra
t_stat_r2, p_value_r2 = ttest_rel(results_df['R2_test'], [0.796] * len(results_df))
print(f"\nPaired t-test cho R² trên tập kiểm tra:")
print(f"t-statistic: {t_stat_r2:.4f}, p-value: {p_value_r2:.4f}")

# So sánh MSE trên tập kiểm tra
t_stat_mse, p_value_mse = ttest_rel(results_df['MSE_test'], [43.1178] * len(results_df))
print(f"\nPaired t-test cho MSE trên tập kiểm tra:")
print(f"t-statistic: {t_stat_mse:.4f}, p-value: {p_value_mse:.4f}")