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
