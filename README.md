📖 Dự án Giữa Kỳ - Môn Xử lý Ngôn ngữ Tự nhiên (NLP) 2024-2025
📝 Giới thiệu
Dự án giữa kỳ môn học Xử lý Ngôn ngữ Tự nhiên (NLP) trong học kỳ năm học 2024-2025. Dự án mô phỏng quy trình xử lý dữ liệu, huấn luyện mô hình học máy, kết hợp với hệ thống khuyến nghị và chatbot. Chatbot được triển khai thông qua API và mô hình học máy được huấn luyện riêng.

Thư mục Chatbox_train: Chứa mã nguồn cho việc huấn luyện chatbot.
Thư mục NguyenTrungHieu22110138: Chứa giao diện và logic xử lý, được viết bằng Django.

🚀 Yêu cầu
Để chạy dự án, cần cài đặt các thư viện sau:
📦 1. Thư viện chuẩn của Python

datetime
os
base64
pickle
json
io.BytesIO

📊 2. Xử lý dữ liệu & trực quan hóa

numpy (as np)
pandas (as pd)
matplotlib.pyplot (as plt)
matplotlib.use('Agg') – để chạy trên server không cần GUI
seaborn (as sns)

🌐 3. Web & Web Scraping

requests
bs4.BeautifulSoup

🌍 4. Django (Framework web Python)

django.http.JsonResponse
django.shortcuts.render
django.views.decorators.csrf.csrf_exempt

📚 5. Xử lý Ngôn ngữ Tự nhiên (NLP)

nltk:
word_tokenize
stopwords
WordNetLemmatizer


nlpaug.augmenter.word (as naw)

🤖 6. Machine Learning - Scikit-learn

Vector hóa văn bản: CountVectorizer, TfidfVectorizer
Mã hóa nhãn: LabelEncoder, LabelBinarizer
Tách tập dữ liệu: train_test_split
Đánh giá mô hình: accuracy_score, classification_report, confusion_matrix
Mô hình:
MultinomialNB
SVC
LogisticRegression
KNeighborsClassifier
DecisionTreeClassifier
RandomForestClassifier
GradientBoostingClassifier
AdaBoostClassifier


Giảm chiều dữ liệu: TruncatedSVD

🧠 7. Machine Learning nâng cao

XGBClassifier (từ xgboost)
Word embeddings: Word2Vec, FastText (từ gensim.models)
Mô hình BERT: BertTokenizer, BertModel (từ transformers)
torch (hỗ trợ BERT)

📐 8. Tiện ích khác

joblib (lưu và truy xuất mô hình ML)
cosine_similarity (tính độ tương đồng vector văn bản)

🛠️ Cài đặt

Cài đặt Python (phiên bản 3.8 trở lên khuyến nghị).
Cài đặt các thư viện cần thiết:pip install django numpy pandas matplotlib seaborn requests beautifulsoup4 nltk nlpaug scikit-learn xgboost gensim transformers torch joblib


Tải các tài nguyên NLTK:import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



▶️ Cách chạy

Mở terminal tại thư mục NguyenTrungHieu22110138 (cùng cấp với file manage.py).
Chạy lệnh:python manage.py runserver


Truy cập đường dẫn cục bộ (ví dụ: http://127.0.0.1:8000) được hiển thị trên terminal.

⚠️ Lưu ý

Các thuật toán AdaBoost và RandomForest đang trong giai đoạn bảo trì.
Tính năng lọc theo nội dung (content-based filtering) cũng đang bảo trì.
Đảm bảo tất cả thư viện đã được cài đặt trước khi chạy.

📧 Liên hệ
Nếu có thắc mắc, vui lòng gửi email đến:Email: hieuanthonydisward@gmail.com

📖 Midterm Project - Natural Language Processing (NLP) 2024-2025
📝 Overview
This is the midterm project for the Natural Language Processing (NLP) course in the 2024-2025 academic semester. The project simulates the pipeline of data processing, machine learning model training, and integrates a recommendation system with a chatbot. The chatbot is implemented using both an API and a machine learning model trained from scratch.

Folder Chatbox_train: Contains the code for training the chatbot.
Folder NguyenTrungHieu22110138: Contains the interface and processing logic, built with Django.

🚀 Requirements
To run the project, ensure the following libraries are installed:
📦 1. Python Standard Library

datetime
os
base64
pickle
json
io.BytesIO

📊 2. Data Processing & Visualization

numpy (as np)
pandas (as pd)
matplotlib.pyplot (as plt)
matplotlib.use('Agg') – for server-side rendering without GUI
seaborn (as sns)

🌐 3. Web & Web Scraping

requests
bs4.BeautifulSoup

🌍 4. Django (Python Web Framework)

django.http.JsonResponse
django.shortcuts.render
django.views.decorators.csrf.csrf_exempt

📚 5. Natural Language Processing (NLP)

nltk:
word_tokenize
stopwords
WordNetLemmatizer


nlpaug.augmenter.word (as naw)

🤖 6. Machine Learning - Scikit-learn

Text vectorization: CountVectorizer, TfidfVectorizer
Label encoding: LabelEncoder, LabelBinarizer
Data splitting: train_test_split
Model evaluation: accuracy_score, classification_report, confusion_matrix
Models:
MultinomialNB
SVC
LogisticRegression
KNeighborsClassifier
DecisionTreeClassifier
RandomForestClassifier
GradientBoostingClassifier
AdaBoostClassifier


Dimensionality reduction: TruncatedSVD

🧠 7. Advanced Machine Learning

XGBClassifier (from xgboost)
Word embeddings: Word2Vec, FastText (from gensim.models)
BERT models: BertTokenizer, BertModel (from transformers)
torch (for BERT support)

📐 8. Utilities

joblib (for saving/loading ML models)
cosine_similarity (for computing text vector similarity)

🛠️ Installation

Install Python (version 3.8 or higher recommended).
Install required libraries:pip install django numpy pandas matplotlib seaborn requests beautifulsoup4 nltk nlpaug scikit-learn xgboost gensim transformers torch joblib


Download NLTK resources:import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



▶️ How to Run

Open a terminal in the NguyenTrungHieu22110138 folder (same level as manage.py).
Run the command:python manage.py runserver


Access the local URL (e.g., http://127.0.0.1:8000) displayed in the terminal.

⚠️ Notes

The AdaBoost and RandomForest algorithms are currently under maintenance.
Content-based filtering is also under maintenance.
Ensure all libraries are installed before running the project.

📧 Contact
For any inquiries, please reach out via email:Email: hieuanthonydisward@gmail.com
