# 📖 Dự án Giữa Kỳ - Môn Xử lý Ngôn ngữ Tự nhiên (NLP) 2024-2025

## 📝 Giới thiệu

Dự án giữa kỳ môn học Xử lý Ngôn ngữ Tự nhiên (NLP) trong học kỳ năm học 2024-2025. Dự án mô phỏng quy trình xử lý dữ liệu, huấn luyện mô hình học máy, kết hợp với hệ thống khuyến nghị và chatbot. Chatbot được triển khai thông qua API và mô hình học máy được huấn luyện riêng.

- **Thư mục** `Chatbox_train`: Chứa mã nguồn cho việc huấn luyện chatbot.
- **Thư mục** `NguyenTrungHieu22110138`: Chứa giao diện và logic xử lý, được viết bằng Django.

## 🚀 Yêu cầu

Để chạy dự án, cần cài đặt các thư viện sau:

### 📦 1. Thư viện chuẩn của Python
- `datetime`
- `os`
- `base64`
- `pickle`
- `json`
- `io.BytesIO`

### 📊 2. Xử lý dữ liệu & trực quan hóa
- `numpy` (as `np`)
- `pandas` (as `pd`)
- `matplotlib.pyplot` (as `plt`)
- `matplotlib.use('Agg')` – để chạy trên server không cần GUI
- `seaborn` (as `sns`)

### 🌐 3. Web & Web Scraping
- `requests`
- `bs4.BeautifulSoup`

### 🌍 4. Django (Framework web Python)
- `django.http.JsonResponse`
- `django.shortcuts.render`
- `django.views.decorators.csrf.csrf_exempt`

### 📚 5. Xử lý Ngôn ngữ Tự nhiên (NLP)
- `nltk`:
  - `word_tokenize`
  - `stopwords`
  - `WordNetLemmatizer`
- `nlpaug.augmenter.word` (as `naw`)

### 🤖 6. Machine Learning - Scikit-learn
- Vector hóa văn bản: `CountVectorizer`, `TfidfVectorizer`
- Mã hóa nhãn: `LabelEncoder`, `LabelBinarizer`
- Tách tập dữ liệu: `train_test_split`
- Đánh giá mô hình: `accuracy_score`, `classification_report`, `confusion_matrix`
- Mô hình:
  - `MultinomialNB`
  - `SVC`
  - `LogisticRegression`
  - `KNeighborsClassifier`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
  - `AdaBoostClassifier`
- Giảm chiều dữ liệu: `TruncatedSVD`

### 🧠 7. Machine Learning nâng cao
- `XGBClassifier` (từ `xgboost`)
- Word embeddings: `Word2Vec`, `FastText` (từ `gensim.models`)
- Mô hình BERT: `BertTokenizer`, `BertModel` (từ `transformers`)
- `torch` (hỗ trợ BERT)

### 📐 8. Tiện ích khác
- `joblib` (lưu và truy xuất mô hình ML)
- `cosine_similarity` (tính độ tương đồng vector văn bản)

## 🛠️ Cài đặt
1. Cài đặt Python (phiên bản 3.8 trở lên khuyến nghị).
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install django numpy pandas matplotlib seaborn requests beautifulsoup4 nltk nlpaug scikit-learn xgboost gensim transformers torch joblib
