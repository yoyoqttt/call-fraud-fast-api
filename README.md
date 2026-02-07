# Fraud Call Detection System - Backend API

A real-time fraud call detection system using Natural Language Processing and Deep Learning to identify and flag potential scam calls with 92% accuracy.

## 🚀 Features

- **Real-time Call Analysis**: Process call transcripts in under 200ms
- **Deep Learning Model**: LSTM networks with attention mechanisms
- **NLP Processing**: TF-IDF vectorization, sentiment analysis, and NER
- **RESTful API**: Fast, asynchronous endpoints built with FastAPI
- **Database Integration**: PostgreSQL for storing call records and fraud patterns
- **Model Versioning**: MLflow for experiment tracking and model management
- **Production Ready**: Docker support, logging, and monitoring

## 📋 Prerequisites

- Python 3.10 or higher
- PostgreSQL 14 or higher
- Redis (optional, for caching)
- Git

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-call-detection-backend.git
cd fraud-call-detection-backend
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### 4. Download NLP Models

```bash
# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 5. Environment Configuration

Create a `.env` file in the root directory:

```env
# Application
APP_NAME=Fraud Call Detection API
APP_VERSION=1.0.0
DEBUG=True
ENVIRONMENT=development

# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/fraud_detection
DB_ECHO=False

# Security
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (Optional)
REDIS_URL=redis://localhost:6379/0

# ML Model
MODEL_PATH=models/fraud_detection_model.h5
CONFIDENCE_THRESHOLD=0.75

# Logging
LOG_LEVEL=INFO
```

### 6. Database Setup

```bash
# Create database
createdb fraud_detection

# Run migrations
alembic upgrade head
```

## 🏃 Running the Application

### Development Mode

```bash
# Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using the provided script
python run.py
```

### Production Mode

```bash
# Using gunicorn with uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

The API will be available at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📁 Project Structure

```
fraud-call-detection-backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration settings
│   │
│   ├── api/                    # API routes
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── fraud_detection.py
│   │   │   ├── calls.py
│   │   │   └── analytics.py
│   │   └── dependencies.py
│   │
│   ├── models/                 # Database models
│   │   ├── __init__.py
│   │   ├── call.py
│   │   └── fraud_pattern.py
│   │
│   ├── schemas/                # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── call.py
│   │   └── prediction.py
│   │
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── fraud_detector.py
│   │   ├── nlp_processor.py
│   │   └── feature_extractor.py
│   │
│   ├── ml/                     # ML models and training
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── preprocessor.py
│   │
│   ├── db/                     # Database configuration
│   │   ├── __init__.py
│   │   ├── session.py
│   │   └── base.py
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── validators.py
│
├── tests/                      # Test files
│   ├── __init__.py
│   ├── test_api.py
│   └── test_fraud_detection.py
│
├── models/                     # Saved ML models
│   ├── .gitkeep
│   └── fraud_detection_model.h5
│
├── data/                       # Data directories
│   ├── raw/
│   ├── processed/
│   └── sample_data/
│
├── logs/                       # Application logs
│
├── alembic/                    # Database migrations
│   └── versions/
│
├── .env                        # Environment variables (not in git)
├── .gitignore
├── requirements.txt
├── README.md
├── Dockerfile
├── docker-compose.yml
└── run.py                      # Application runner
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_api.py -v
```

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down
```

### Build Docker Image Manually

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 --env-file .env fraud-detection-api
```

## 📊 API Endpoints

### Fraud Detection

- `POST /api/v1/detect` - Analyze a call transcript for fraud
- `POST /api/v1/detect/batch` - Batch process multiple calls
- `GET /api/v1/calls/{call_id}` - Get call details
- `GET /api/v1/calls` - List all calls with pagination

### Analytics

- `GET /api/v1/analytics/summary` - Get fraud detection summary
- `GET /api/v1/analytics/patterns` - Get fraud patterns
- `GET /api/v1/analytics/trends` - Get fraud trends over time

### Health Check

- `GET /health` - Health check endpoint
- `GET /` - Root endpoint with API information

## 🔧 Model Training

To train a new fraud detection model:

```bash
# Train model with default parameters
python -m app.ml.train

# Train with custom parameters
python -m app.ml.train --epochs 50 --batch-size 64 --learning-rate 0.001
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `SECRET_KEY` | JWT secret key | Required |
| `MODEL_PATH` | Path to trained model | `models/fraud_detection_model.h5` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for fraud detection | `0.75` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `REDIS_URL` | Redis connection URL | Optional |

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration
- Rate limiting (recommended in production)
- Input validation with Pydantic

## 📈 Performance

- Average response time: <200ms
- Supports concurrent requests
- Async database operations
- Redis caching for frequent queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Shivam Srivastava**
- Email: shivasrivastava451@gmail.com
- LinkedIn: [linkedin.com/in/shivam-srivastava](https://linkedin.com/in/shivam-srivastava)
- GitHub: [@yourusername](https://github.com/yoyoqttt)

## 🙏 Acknowledgments

- FastAPI framework
- TensorFlow and PyTorch teams
- NLTK and spaCy communities
- All contributors and supporters

## 📞 Support

For support, email shivasrivastava451@gmail.com or create an issue in the repository.

