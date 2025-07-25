import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

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

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
rf_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = rf_model.predict(X_test)

y_train_pred = rf_model.predict(X_train)
print(f'R² trên tập huấn luyện: {r2_score(y_train, y_train_pred):.4f}')
print(f'MSE trên tập huấn luyện: {mean_squared_error(y_train, y_train_pred):.4f}')

##Dùng cross-validation để đánh giá mô hình
#from sklearn.model_selection import cross_val_score
#cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
#cv_scores_mse = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
#print(f'R² cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
#print(f'MSE cross-validation: {-cv_scores_mse.mean():.4f} ± {cv_scores_mse.std():.4f}')

##Tính toán Feature Importances
#importances = rf_model.feature_importances_
#feature_names = X.columns
#feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
#feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
#print(feature_importance_df)

## Đánh giá mô hình bằng R² và MSE
#r2 = r2_score(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)

## In kết quả
#print(f'R²: {r2:.4f}')     #R² : 0.7960
#print(f'MSE: {mse:.4f}')   #MSE: 43.1778