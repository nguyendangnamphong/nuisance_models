# nuisance_models              

          
**1) Random Forest**              

 **a) Thư viện Scikit‑learn**         
 *Lịch sử:*             
 + Ra đời năm 2007 dưới tên scikits.learn, do David Cournapeau phát triển trong chương trình Google Summer of Code, và chính thức ra mắt version 0.1 vào năm 2010            
 + Được viết chủ yếu bằng Python, tích hợp chặt với NumPy và SciPy; các thuật toán cốt lõi được tối ưu hóa bằng Cython, và áp dụng thư viện như LIBSVM/LINEAR              
 + Phát hành theo giấy phép BSD, cho phép sử dụng rộng rãi cho cả nghiên cứu và thương mại                                    

*Tính năng chính:*
+ Hỗ trợ thuật toán học máy: phân lớp và hồi quy (SVM, logistic, random forests, gradient boosting), phân cụm (k-means, DBSCAN), giảm chiều (PCA, LDA).                            
+ Xử lý dữ liệu: scaling, encoding, chọn đặc trưng, xóa thiếu dữ liệu, ....                                              
                                                  
**b) Random Forest**                     
  *- Tổng quan:* là một kỹ thuật học máy tổng hợp (ensemble learning) mạnh mẽ, kết hợp nhiều decision trees để cải thiện độ chính xác và giảm hiện tượng overfitting.                                      
  *- Cách hoạt động:*                                             
  + Bagging: Tạo nhiều tập con dữ liệu học bằng phương pháp bootstrap (lấy mẫu có hoàn lại). Mỗi cây trong rừng được huấn luyện trên một tập dữ liệu khác nhau để tăng sự đa dạng và giảm tương quan giữa các cây.
  + Random feature selection: Khi chia node, mỗi cây chỉ xem xét một tập con ngẫu nhiên của các đặc trưng thay vì hết tất cả. Giúp tránh một vài đặc trưng nổi bật áp đảo, tạo sự không đồng nhất giữa các cây
  + Aggregation: Classification (lấy phiếu đa số từ các cây) ; Regression (lấy giá trị trung bình đầu ra của các cây)
  + Out-Of-Bag: Dữ liệu không nằm trong bootstrap dùng để đánh giá mô hình mà không cần dùng cross-validation riêng. Đây là ước lượng sai số nội bộ và khá đáng tin cậy                    
**c) Kết quả đo lường**                   
+ *R²*: 0.796
+ *MSE*: 43.1178
+ *R² cross-validation*: -1.3895 ± 1.0452                            
+ *MSE cross-validation*: 493.3883 ± 595.7211
+ *R² trên tập huấn luyện*(kiểm tra overfiting): 0.8996
+ *MSE trên tập huấn luyện*(kiểm tra overfiting): 23.1840                                                  
**d) Đánh giá dựa trên kết quả đo lường**                                                              
*Hiệu suất trên tập kiểm tra*
+ R² = 0.796, cho thấy mô hình giải thích được 79.6% biến thiên của độ dẫn điện, một kết quả khá tốt, thể hiện khả năng dự đoán tốt trên tập dữ liệu kiểm tra.                                
+ MSE = 43.1178, là sai số trung bình bình phương, cho thấy mức sai số dự đoán tương đối thấp, nhưng cần so sánh với thang đo của độ dẫn điện (thường dao động từ 4% đến 100% IACS, nên MSE này là chấp nhận được).
                                                       
*Cross-validation*
+ R² cross-validation = -1.3895 ± 1.0452, một giá trị âm cho thấy mô hình có hiệu suất kém hơn so với dự đoán trung bình (horizontal line) trên các tập con, với độ lệch chuẩn lớn (1.0452), cho thấy sự không ổn định.                                                                    
+ MSE cross-validation = 493.3883 ± 595.7211, giá trị cao và độ lệch chuẩn lớn (595.7211) cho thấy mô hình không tổng quát hóa tốt, có thể do overfitting hoặc dữ liệu không đủ đa dạng.   
   
*Kiểm tra overfitting*
+ R² trên tập huấn luyện = 0.8996, cao hơn đáng kể so với R² trên tập kiểm tra (0.796), cho thấy dấu hiệu overfitting.                            
+ MSE trên tập huấn luyện = 23.1840, thấp hơn nhiều so với MSE trên tập kiểm tra (43.1178), củng cố giả thuyết về overfitting.
                                                         
**e) Đánh giá chung**                                                     
*Độ phù hợp với DML*
  + Mô hình hiện tại có R² trên tập kiểm tra (0.796) khá tốt, cho thấy khả năng dự đoán tốt trên dữ liệu mới, phù hợp với yêu cầu của nuisance model trong DML. Tuy nhiên, kết quả cross-validation (R² = -1.3895) cho thấy khả năng tổng quát hóa kém, có thể gây bias trong ước lượng nhân quả.                                              
  + Nguyên nhân: DML sử dụng nuisance models để ước lượng các hàm trung bình có điều kiện, như E[Y|X] (dự đoán biến kết quả dựa trên các đặc điểm) và E[T|X] (dự đoán biến điều trị dựa trên các đặc điểm). Random Forest hiện tại có thể làm tốt bước này trên tập kiểm tra, nhưng cross-validation kém cho thấy nó có thể không tổng quát hóa tốt trên dữ liệu mới, dẫn đến sai lệch trong bước cuối cùng của DML (ước lượng hiệu ứng nhân quả).
                                                                     
*Độ phù hợp với bộ dữ liệu*
  + Mô hình có độ phù hợp trung bình, nhờ khả năng xử lý dữ liệu cao chiều (32 đặc điểm) và tầm quan trọng của các đặc điểm như Cu (0.495574), Aging_Time_h (0.161614), Ti (0.115228). Đặc điểm này phát huy nhờ Random Forest có thể mô hình hóa mối quan hệ phi tuyến giữa thành phần hóa học, điều kiện xử lý, và độ dẫn điện, phù hợp với dữ liệu phức tạp.
 + Cách phát huy: Tinh chỉnh tham số (Sử dụng GridSearchCV để tối ưu hóa n_estimators, max_depth, min_samples_split, giảm overfitting), Giảm chiều dữ liệu(Bỏ các đặc điểm có tầm quan trọng bằng 0).                                            
**2) Gradient Boosting**


