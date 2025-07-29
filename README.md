# nuisance_models              

          
 **1) Thư viện Scikit‑learn**         
 *Lịch sử:*             
 + Ra đời năm 2007 dưới tên scikits.learn, do David Cournapeau phát triển trong chương trình Google Summer of Code, và chính thức ra mắt version 0.1 vào năm 2010            
 + Được viết chủ yếu bằng Python, tích hợp chặt với NumPy và SciPy; các thuật toán cốt lõi được tối ưu hóa bằng Cython, và áp dụng thư viện như LIBSVM/LINEAR              
 + Phát hành theo giấy phép BSD, cho phép sử dụng rộng rãi cho cả nghiên cứu và thương mại                                    

*Tính năng chính:*
+ Hỗ trợ thuật toán học máy: phân lớp và hồi quy (SVM, logistic, random forests, gradient boosting), phân cụm (k-means, DBSCAN), giảm chiều (PCA, LDA).                            
+ Xử lý dữ liệu: scaling, encoding, chọn đặc trưng, xóa thiếu dữ liệu, ....                                              
                                                  
**2) Random Forest**                     
  *- Tổng quan:* là một kỹ thuật học máy tổng hợp (ensemble learning) mạnh mẽ, kết hợp nhiều decision trees để cải thiện độ chính xác và giảm hiện tượng overfitting.                                      
  *- Cách hoạt động:*                                             
  + Bagging: Tạo nhiều tập con dữ liệu học bằng phương pháp bootstrap (lấy mẫu có hoàn lại). Mỗi cây trong rừng được huấn luyện trên một tập dữ liệu khác nhau để tăng sự đa dạng và giảm tương quan giữa các cây.
  + Random feature selection: Khi chia node, mỗi cây chỉ xem xét một tập con ngẫu nhiên của các đặc trưng thay vì hết tất cả. Giúp tránh một vài đặc trưng nổi bật áp đảo, tạo sự không đồng nhất giữa các cây
  + Aggregation: Classification (lấy phiếu đa số từ các cây) ; Regression (lấy giá trị trung bình đầu ra của các cây)
  + Out-Of-Bag: Dữ liệu không nằm trong bootstrap dùng để đánh giá mô hình mà không cần dùng cross-validation riêng. Đây là ước lượng sai số nội bộ và khá đáng tin cậy


**3) Kết quả đo lường**
                  
**a) Tập cố định**                  
+ *R²*: 0.796
+ *MSE*: 43.1178
+ *R² cross-validation*: -1.3895 ± 1.0452                            
+ *MSE cross-validation*: 493.3883 ± 595.7211
+ *R² trên tập huấn luyện*(kiểm tra overfiting): 0.8996
+ *MSE trên tập huấn luyện*(kiểm tra overfiting): 23.1840

                                             
**b) Đánh giá dựa trên kết quả đo lường ở tập cố định**                                                              
*Hiệu suất trên tập kiểm tra*
+ R² = 0.796, cho thấy mô hình giải thích được 79.6% biến thiên của độ dẫn điện, một kết quả khá tốt, thể hiện khả năng dự đoán tốt trên tập dữ liệu kiểm tra.                                
+ MSE = 43.1178, là sai số trung bình bình phương, cho thấy mức sai số dự đoán tương đối thấp, nhưng cần so sánh với thang đo của độ dẫn điện (thường dao động từ 4% đến 100% IACS, nên MSE này là chấp nhận được).
                                                       
*Cross-validation*
+ R² cross-validation = -1.3895 ± 1.0452, một giá trị âm cho thấy mô hình có hiệu suất kém hơn so với dự đoán trung bình (horizontal line) trên các tập con, với độ lệch chuẩn lớn (1.0452), cho thấy sự không ổn định.                                                                    
+ MSE cross-validation = 493.3883 ± 595.7211, giá trị cao và độ lệch chuẩn lớn (595.7211) cho thấy mô hình không tổng quát hóa tốt, có thể do overfitting hoặc dữ liệu không đủ đa dạng.   
   
*Kiểm tra overfitting*
+ R² trên tập huấn luyện = 0.8996, cao hơn đáng kể so với R² trên tập kiểm tra (0.796), cho thấy dấu hiệu overfitting.                            
+ MSE trên tập huấn luyện = 23.1840, thấp hơn nhiều so với MSE trên tập kiểm tra (43.1178), củng cố giả thuyết về overfitting.
                                                         
**c) Tập ngẫu nhiên**                                                             
Kết quả cho các random_state khác nhau:                                                                                                  
                                  
<img width="1862" height="889" alt="Capture" src="https://github.com/user-attachments/assets/debf3f0f-9600-4ab5-a47a-25098365c16b" />                                                    
<img width="1170" height="305" alt="image" src="https://github.com/user-attachments/assets/67d2ee0c-0ec3-4f79-96f5-310c4df0f3fa" />                                                      

**d) Đánh giá dựa trên kết quả đo lường ở tập ngẫu nhiên**                                                
*Biến động hiệu suất*                                         
+ R² trên tập kiểm tra dao động từ 0.796 (random_state=42) đến 0.8537 (random_state=456), và MSE từ 34.8817 đến 43.1778. Sự biến động này cho thấy hiệu suất mô hình phụ thuộc phần nào vào cách chia dữ liệu, nhưng không quá lớn.                                                                                                                
+ Trên tập huấn luyện, R² dao động từ 0.8890 đến 0.8996 và MSE từ 23.1840 đến 24.8907, cho thấy hiệu suất trên tập huấn luyện khá ổn định và luôn tốt hơn tập kiểm tra.


*Kiểm tra t-test*                                            
+ p-value cho cả R² (0.2022) và MSE (0.2030) đều lớn hơn 0.05, nghĩa là không có sự khác biệt có ý nghĩa thống kê giữa các lần chạy với random_state khác nhau. Điều này cho thấy mô hình tương đối ổn định khi thay đổi cách chia dữ liệu.

                                                        
**e) Kết luận**                        
*Dấu hiệu overfiting*                             
+ Sự chênh lệch giữa hiệu suất trên tập huấn luyện (R² ~0.89–0.90, MSE ~23–25) và tập kiểm tra (R² ~0.80–0.85, MSE ~34–43).                               
+ Kết quả cross-validation rất kém (R² âm, MSE cao), cho thấy khả năng tổng quát hóa hạn chế.

                                                          
*Mức độ overfiting*                                                                          
+ Sự chênh lệch giữa tập huấn luyện và kiểm tra không quá lớn (R² chênh ~0.05–0.10, MSE chênh ~10–20), cho thấy overfitting ở mức nhẹ.                                     
+ Hiệu suất trên tập kiểm tra vẫn khá tốt (R² trung bình ~0.82), và mô hình ổn định khi thay đổi random_state (p-value > 0.05), nên không phải là overfitting nghiêm trọng.                                     
                                                                                                               



