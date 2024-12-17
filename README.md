# IMDB Sentiment Analysis Project

## 🎯 **Project Overview**
This project implements a machine learning pipeline to classify IMDB movie reviews into Positive or Negative sentiments. The pipeline includes data preprocessing, model training, evaluation, and a web-based interface for predictions. The project is fully containerized using Docker and uses Neptune.ai for experiment tracking and visualization.

---

## 🚀 **Features**
1. **Data Preprocessing**: Cleaning and vectorizing movie reviews using TF-IDF.
2. **Model Training**: Logistic Regression with hyperparameter tuning using Random Search.
3. **Evaluation Metrics**: Accuracy, F1-score, Confusion Matrix, and ROC-AUC curve.
4. **Neptune.ai Integration**: Logs experiments, metrics, and visualizations.
5. **Web Interface**: Simple frontend (HTML, CSS, JS) for users to input reviews and get predictions.
6. **Containerization**: Backend and frontend are containerized with Docker and orchestrated using Docker Compose.

---

## 🛠️ **Technologies Used**
- **Python 3.9**: Main programming language.
- **FastAPI**: Backend framework for serving predictions.
- **Scikit-learn**: ML library for Logistic Regression and TF-IDF.
- **Neptune.ai**: For experiment tracking.
- **Docker & Docker Compose**: Containerization of the application.
- **HTML, CSS, JavaScript**: Frontend interface.
- **Matplotlib & Seaborn**: Visualization tools.
- **Pandas & NumPy**: Data handling and processing.

---

## 📂 **Project Structure**
```
IMDB-Review-Classifier/
│
├── data/
│   ├── raw/                 # Raw dataset (IMDB Dataset.csv)
│   └── processed/           # Processed TF-IDF data and labels
│       ├── X_train_tfidf.npz
│       ├── X_test_tfidf.npz
│       ├── y_train.csv
│       ├── y_test.csv
│       └── tfidf_vectorizer.pkl
│
├── model/
│   └── best_sentiment_model.pkl   # Trained Logistic Regression model
│
├── backend/
│   ├── requirements.txt       # Python dependencies
│   ├── api.py                 # FastAPI backend for predictions
│   ├── train_model.py         # Model training with Random Search and Neptune logging
│   └── data_processing.py     # Data cleaning and TF-IDF processing
│
├── frontend/
│   ├── index.html             # Web interface
│   ├── style.css              # Styling for the web interface
│   └── static/                # Static assets like images (snowflakes, icons)
│
├── docker/
│   ├── Dockerfile.backend     # Dockerfile for the backend
│   ├── Dockerfile.frontend    # Dockerfile for the frontend
│   └── docker-compose.yml     # Docker Compose configuration
│
└── README.md                  # Project documentation
```

---

## 📊 **Dataset**
- Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Size: 50,000 rows (Positive and Negative reviews).
- Format: CSV

---

## ⚙️ **Setup Instructions**

### 1. **Clone the repository**
```bash
git clone https://github.com/himarygr/IMDB-Review-Classifier.git
cd IMDB-Review-Classifier
```

### 2. **Install dependencies** (for local development)
#### Backend
```bash
cd backend
pip install -r requirements.txt
```
#### Frontend
No installation is required for the static frontend.

### 3. **Data Preprocessing**
Run the following script to clean and vectorize data:
```bash
python backend/data_processing.py
```

### 4. **Model Training**
Run model training with Random Search and log metrics to Neptune.ai:
```bash
python backend/train_model.py
```

### 5. **Run with Docker Compose**
To build and run the project using Docker:
```bash
cd docker
docker-compose up --build
```
- Backend will run on: `http://localhost:8000`
- Frontend will run on: `http://localhost:8501`

---

## 🔗 **Endpoints** (Backend API)
| Method | Endpoint       | Description               |
|--------|----------------|---------------------------|
| POST   | `/predict/`    | Predict sentiment of a review |

**Example Request:**
```json
{
  "review": "The movie was absolutely fantastic! Great acting and direction."
}
```
**Example Response:**
```json
{
  "sentiment": "positive"
}
```

---

## 🖥️ **Web Interface**
The frontend provides a simple interface where users can:
1. Enter a movie review.
2. Click the **"Analyze Sentiment"** button.
3. See whether the review is classified as Positive 😊 or Negative 😞.

---

## 🧪 **Experiment Tracking**
All experiments, metrics, and visualizations are logged to **Neptune.ai**.

### **Logged Items:**
1. **Hyperparameters**: `C`, `solver`, `max_iter`.
2. **Metrics**: Accuracy, F1-score.
3. **Confusion Matrix**: Uploaded as an image.
4. **ROC-AUC Curve**: Uploaded as an image.
5. **CPU & Memory Usage**: System resource monitoring.

---

## 🎨 **Visualizations in Neptune.ai**
- Confusion Matrix
- ROC-AUC Curve
- Accuracy and F1-Score
- Hyperparameter values
- CPU/Memory usage during training

---

## 🔮 **Future Improvements**
- Add more classifiers (e.g., SVM, Random Forest) for comparison.
- Integrate Grid Search for exhaustive hyperparameter tuning.
- Deploy the project to a cloud service (AWS, GCP, etc.).
- Enhance the frontend with a modern framework (React or Vue.js).

---

## 🤝 **Contributing**
Feel free to fork the repository, create a branch, and submit pull requests for new features or bug fixes!

---

## 📜 **License**
This project is licensed under the MIT License.

---

## 📞 **Contact**
For any questions or suggestions:
- **Email**: lilley@ya.ru
- **GitHub**: [himarygr](https://github.com/himarygr)
