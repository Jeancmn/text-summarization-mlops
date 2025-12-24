# 📝 Text Summarization MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.78.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-latest-yellow.svg)](https://huggingface.co/transformers/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Production-ready MLOps pipeline for text summarization using Hugging Face Transformers (PEGASUS), featuring automated training, evaluation, and deployment via FastAPI.**

---

## 🎯 Overview

This project implements a complete end-to-end Machine Learning Operations (MLOps) pipeline for text summarization using state-of-the-art NLP models from Hugging Face. Built with a modular architecture following software engineering best practices, it demonstrates the full lifecycle of ML model development—from data ingestion to production deployment.

### ✨ Key Features

- 🤖 **Pre-trained Model Fine-tuning**: Google PEGASUS (CNN/DailyMail) adapted to SAMSum dataset
- 🔄 **Automated 4-Stage Pipeline**: Data Ingestion → Transformation → Training → Evaluation
- 🚀 **FastAPI REST API**: Production-ready inference and training endpoints
- 📊 **Model Evaluation**: ROUGE and BLEU metrics for performance tracking
- 🐳 **Docker Support**: Containerized deployment for scalability
- 📝 **Structured Logging**: Comprehensive logging throughout the pipeline
- ⚙️ **YAML Configuration**: Centralized parameter management
- 🏗️ **Modular Architecture**: Clean separation of concerns for maintainability

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                         │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  /train endpoint │         │ /predict endpoint │             │
│  └────────┬─────────┘         └────────┬──────────┘             │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
            ▼                              ▼
  ┌─────────────────────┐       ┌──────────────────────┐
  │  Training Pipeline  │       │  Prediction Pipeline │
  └─────────┬───────────┘       └──────────┬───────────┘
            │                              │
   ┌────────┴────────┐                     │
   ▼                 ▼                     ▼
┌──────┐  ┌──────────────────┐  ┌──────────────────┐
│Stage1│  │     Stage 2      │  │  Trained Model   │
│Data  │─▶│  Transformation  │─▶│  + Tokenizer     │
│Ingest│  │   (Tokenization) │  │                  │
└──────┘  └──────────────────┘  └──────────────────┘
   │             │
   ▼             ▼
┌──────┐  ┌──────────────────┐  ┌──────────────────┐
│Stage3│  │     Stage 4      │  │  Evaluation      │
│Model │─▶│   Evaluation     │─▶│  Metrics (CSV)   │
│Trainer  │  (ROUGE, BLEU)   │  │                  │
└──────┘  └──────────────────┘  └──────────────────┘
```

---

## 📁 Project Structure

```bash
text-summarization-mlops/
│
├── app.py                      # FastAPI application with endpoints
├── main.py                     # Training pipeline orchestrator
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── setup.py                    # Package installation script
├── params.yaml                 # Training hyperparameters
│
├── config/
│   └── config.yaml            # Pipeline configuration (paths, models)
│
├── src/textSummarizer/
│   ├── components/            # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/              # Orchestration pipelines
│   │   ├── stage_1_data_ingestion_pipeline.py
│   │   ├── stage_2_data_transformation_pipeline.py
│   │   ├── stage_3_model_trainer_pipeline.py
│   │   ├── stage_4_model_evaluation.py
│   │   └── predicition_pipeline.py
│   │
│   ├── config/                # Configuration management
│   │   └── configuration.py
│   │
│   ├── entity/                # Data classes and schemas
│   ├── constants/             # Project constants
│   ├── logging/               # Logging utilities
│   └── utils/                 # Helper functions
│
├── research/                  # Jupyter notebooks for experimentation
└── artifacts/                 # Generated during training (models, data)
    ├── data_ingestion/
    ├── data_transformation/
    ├── model_trainer/
    └── model_evaluation/
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- (Optional) Docker for containerized deployment

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jeancmn/text-summarization-mlops.git
   cd text-summarization-mlops
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode**
   ```bash
   pip install -e .
   ```

---

## 💻 Usage

### 🎓 Training the Model

Run the complete training pipeline (all 4 stages):

```bash
python main.py
```

This will execute:
1. **Data Ingestion**: Download and extract SAMSum dataset
2. **Data Transformation**: Tokenize dialogues using PEGASUS tokenizer
3. **Model Training**: Fine-tune PEGASUS on SAMSum conversations
4. **Model Evaluation**: Calculate ROUGE and BLEU scores

Training artifacts will be saved in `artifacts/` directory.

---

### 🌐 Running the API Server

Start the FastAPI server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at: **http://localhost:8080**

Interactive API documentation: **http://localhost:8080/docs**

---

### 📡 API Endpoints

#### 1. **Root Endpoint**
```http
GET /
```
Redirects to interactive API documentation.

#### 2. **Training Endpoint**
```http
GET /train
```
Triggers the complete training pipeline.

**Response:**
```json
"Training successful !!"
```

#### 3. **Prediction Endpoint**
```http
POST /predict
```

**Request Body:**
```json
{
  "text": "Your long dialogue or text to summarize goes here..."
}
```

**Response:**
```json
{
  "summary": "Concise summary of the input text"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Long conversation text here..."}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8080/predict",
    json={"text": "Your dialogue text here..."}
)
print(response.json())
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t text-summarizer:latest .
```

### Run Container

```bash
docker run -p 8080:8080 text-summarizer:latest
```

Access the API at **http://localhost:8080**

---

## ⚙️ Configuration

### `config/config.yaml`

Defines pipeline stages, model checkpoints, and data paths:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/samsum_dataset
  tokenizer_name: google/pegasus-cnn_dailymail

model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation/samsum_dataset
  model_ckpt: google/pegasus-cnn_dailymail

model_evaluation:
  root_dir: artifacts/model_evaluation
  data_path: artifacts/data_transformation/samsum_dataset
  model_path: artifacts/model_trainer/pegasus-samsum-model
  tokenizer_path: artifacts/model_trainer/tokenizer
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

### `params.yaml`

Training hyperparameters:

```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1e6
  gradient_accumulation_steps: 16
```

Adjust these parameters based on your computational resources and desired model performance.

---

## 📊 Model Evaluation

The pipeline automatically evaluates the fine-tuned model using:

- **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L): Measure overlap with reference summaries
- **BLEU Score**: Evaluate translation/generation quality

Metrics are saved in `artifacts/model_evaluation/metrics.csv`.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **NLP Framework** | 🤗 Hugging Face Transformers |
| **Model** | Google PEGASUS (CNN/DailyMail) |
| **API Framework** | FastAPI, Uvicorn |
| **ML Framework** | PyTorch |
| **Data Processing** | Pandas, NLTK |
| **Evaluation** | SacreBLEU, ROUGE Score |
| **Configuration** | PyYAML, python-box |
| **Containerization** | Docker |
| **Logging** | Custom logging module |

---

## 📈 Pipeline Stages Explained

### **Stage 1: Data Ingestion**
- Downloads SAMSum dataset (conversational dialogue dataset)
- Extracts and organizes data into artifacts directory
- Validates data integrity

### **Stage 2: Data Transformation**
- Tokenizes dialogues and summaries using PEGASUS tokenizer
- Prepares input tensors for model training
- Handles text preprocessing (truncation, padding)

### **Stage 3: Model Training**
- Fine-tunes pre-trained PEGASUS model on SAMSum
- Implements gradient accumulation for memory efficiency
- Saves model checkpoints and tokenizer

### **Stage 4: Model Evaluation**
- Generates predictions on test set
- Calculates ROUGE and BLEU metrics
- Exports evaluation results to CSV

---

## 🔍 Example Use Case

**Input (Dialogue):**
```
Person A: Hey, are you free this weekend?
Person B: Yeah, I think so. Why?
Person A: I was thinking we could go hiking at the national park.
Person B: That sounds great! What time?
Person A: How about 8 AM on Saturday?
Person B: Perfect, I'll bring some snacks.
```

**Output (Summary):**
```
Person A suggests going hiking at the national park this weekend. 
Person B agrees and offers to bring snacks. They plan to meet at 8 AM on Saturday.
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Future Enhancements

- [ ] Add support for multiple models (BART, T5, etc.)
- [ ] Implement MLflow for experiment tracking
- [ ] Add Kubernetes deployment configurations
- [ ] Create batch prediction endpoint
- [ ] Implement model versioning
- [ ] Add unit and integration tests
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Add monitoring and observability (Prometheus, Grafana)

---

## 📄 License

---

## 👨‍💻 Author

**Jean Mangones Nardey**

- GitHub: [@Jeancmn](https://github.com/Jeancmn)
- LinkedIn: [jeanm-nardey](https://www.linkedin.com/in/jeanm-nardey/)
- Email: nardeyjean@gmail.com

---

## 📖 Deep Dive: How It Works & Why It Matters

### 🔍 **How Does the Pipeline Work?**

#### **Training Mode: The Complete Journey**

```
User executes: python main.py
         ↓
┌────────────────────────────────────────────┐
│  STAGE 1: Data Ingestion                  │
│  - Downloads SAMSum dataset from GitHub    │
│  - Extracts 14,732 conversations           │
│  - Validates data integrity                │
└─────────────┬──────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│  STAGE 2: Data Transformation              │
│  - Tokenizes dialogues (text → numbers)    │
│  - Applies PEGASUS tokenizer               │
│  - Prepares PyTorch tensors                │
└─────────────┬──────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│  STAGE 3: Model Training                   │
│  - Fine-tunes pre-trained PEGASUS          │
│  - 1 epoch with gradient accumulation      │
│  - Saves model + tokenizer checkpoints     │
└─────────────┬──────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│  STAGE 4: Model Evaluation                 │
│  - Generates predictions on test set       │
│  - Calculates ROUGE-1, ROUGE-2, ROUGE-L    │
│  - Calculates BLEU score                   │
│  - Exports metrics to CSV                  │
└────────────────────────────────────────────┘
```

**Why This Sequential Order?**
- **Ingestion First**: No data = no training
- **Transformation Before Training**: Models need numerical tensors, not raw text
- **Evaluation Last**: Measures if fine-tuning improved the base model

#### **Production Mode: Real-Time Inference**

```
User sends POST /predict with text
         ↓
┌────────────────────────────────────────────┐
│  FastAPI Receives Request                  │
│  - Validates JSON format                   │
│  - Extracts input text                     │
└─────────────┬──────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│  PredictionPipeline                        │
│  - Loads fine-tuned model                  │
│  - Loads tokenizer                         │
│  - Sets generation config                  │
│    (length_penalty=0.8, num_beams=8)       │
└─────────────┬──────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│  Model Inference                           │
│  - Tokenizes input text                    │
│  - Generates token sequence                │
│  - Decodes to human-readable text          │
└─────────────┬──────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│  Response to User                          │
│  - Returns JSON with summary               │
│  - Typical latency: 2-5 seconds            │
└────────────────────────────────────────────┘
```

---

### 🎯 **What Problem Does This Solve? (Real-World Impact)**

#### **The Daily Information Overload Problem**

**Modern professionals face:**
- Hundreds of long emails daily
- Multi-hour meeting transcriptions
- Endless chat conversations
- Reports and articles requiring hours to read

**This Project's Solution:**
- ✅ **Time Savings**: Read 30-second summary vs. 10-minute document
- ✅ **Faster Decision-Making**: Identify key points without full reading
- ✅ **Increased Productivity**: Process more information in less time
- ✅ **Better Focus**: Spend time on what truly matters

#### **Technical Purpose: Beyond Just ML**

This isn't just a model—it's a **complete MLOps demonstration** showing:

1. **Full ML Production Cycle**
   - Raw data → Trained model → Deployed API
   - Not just experimental notebooks

2. **Scalable Architecture**
   - Modular, maintainable code
   - Configuration separated from logic
   - Structured logging for production debugging

3. **Reproducibility**
   - Anyone can clone and run
   - Version-controlled configs (YAML)
   - Consistent environments (Docker)

4. **Software Engineering Best Practices**
   - Separation of concerns (independent pipelines)
   - Configuration management
   - RESTful API standards
   - Comprehensive documentation

---

### 🧠 **Key Technical Decisions Explained**

#### **Why PEGASUS?**
- **Pre-trained for summarization** (Gap Sentence Generation objective)
- **State-of-the-art results** in 2020 benchmarks
- **Optimal balance** between quality and inference speed
- **Efficient fine-tuning** (requires less data than training from scratch)

#### **Why SAMSum Dataset?**
- **14,732 real conversations** from messaging platforms
- **Human-written summaries** (high-quality ground truth)
- **Natural, casual language** (realistic use cases)
- **Dialogue format** (different from news articles, more challenging)

#### **Why FastAPI over Flask/Django?**
- ✅ **Automatic API documentation** (`/docs` endpoint)
- ✅ **Data validation** built-in (fewer runtime errors)
- ✅ **Async support** (handles concurrent requests efficiently)
- ✅ **Modern Python** (type hints, async/await)
- ✅ **High performance** (comparable to Node.js/Go)

---

## 💡 **Key Learnings & Insights**

### **1. MLOps ≠ Just Machine Learning**

**This project proves that production ML requires:**

| Component | What It Means |
|-----------|---------------|
| **Automated Pipelines** | No manual script execution |
| **API Layer** | Models must be consumable by applications |
| **Config Management** | Easy parameter tuning without code changes |
| **Systematic Evaluation** | Trackable, reproducible metrics |
| **Production Logging** | Debug issues in deployed systems |

**📌 Key Insight:** 90% of real ML work is infrastructure, not the model itself.

---

### **2. Modular Architecture = Maintainable Code**

Separating components (`data_ingestion`, `model_trainer`, etc.) enables:
- ✅ **Independent testing** of each stage
- ✅ **Modification without breaking** other parts
- ✅ **Code reusability** across projects
- ✅ **Team collaboration** without merge conflicts

**📌 Key Insight:** Clean code matters as much in ML as in traditional software engineering.

---

### **3. Transfer Learning Democratizes ML**

**Comparison:**
- **Training PEGASUS from scratch**: Millions of examples, weeks of GPU time, $$$
- **Fine-tuning PEGASUS here**: 14K examples, ~1 hour on basic GPU, $

**📌 Key Insight:** You don't need Google-scale resources to build production ML solutions.

---

### **4. Evaluation Metrics Guide, But Humans Decide**

#### **ROUGE Scores Explained:**
- **ROUGE-1**: Individual word matches (lexical overlap)
- **ROUGE-2**: Two-word phrase matches (bigrams)
- **ROUGE-L**: Longest common subsequence (structural similarity)

#### **Typical Benchmarks:**
- ROUGE-1 > 0.40 → Model captures key terms
- ROUGE-2 > 0.20 → Maintains coherent phrases
- ROUGE-L > 0.35 → Preserves logical structure

**📌 Key Insight:** Metrics provide guidance, but **human evaluation** (Is this summary useful?) is the ultimate test.

---

### **5. Why Configuration Files Matter**

**Separation of Config (`config.yaml`, `params.yaml`) from Code:**
- ✅ **Easy experimentation** (change hyperparameters without touching code)
- ✅ **Version control** (track what config produced which results)
- ✅ **Environment flexibility** (dev/staging/prod configs)
- ✅ **Reproducibility** (anyone can recreate your results)

**📌 Key Insight:** Good config management is a hallmark of mature ML systems.

---

## 🎓 **What This Project Demonstrates (For Your Career)**

### **For Recruiters/Hiring Managers:**

This project proves the candidate can:

1. ✅ **End-to-End ML**: Data → Model → Deployment (not just notebooks)
2. ✅ **Production Mindset**: APIs, logging, configs, Docker
3. ✅ **Modern Stack**: Transformers, FastAPI, PyTorch
4. ✅ **Clean Code**: Modular architecture, separation of concerns
5. ✅ **Documentation**: Professional README, clear structure

**Translation:** This person understands **MLOps**, not just ML theory.

---

### **Technical Skills Showcased:**

| Skill Category | Evidence in Project |
|----------------|---------------------|
| **NLP/Deep Learning** | PEGASUS fine-tuning, tokenization, attention mechanisms |
| **MLOps** | Automated pipelines, model evaluation, versioning |
| **API Development** | FastAPI with prediction & training endpoints |
| **DevOps** | Docker containerization, environment management |
| **Data Engineering** | ETL pipeline (ingestion → transformation) |
| **Software Engineering** | Modular design, config management, logging |

---

## 🚀 **Real-World Applications**

This same architecture can be adapted for:

1. **Customer Support**: Summarize long support tickets
2. **Legal/Healthcare**: Condense lengthy documents
3. **News Aggregation**: Auto-generate headlines
4. **Meeting Notes**: Transcription → Executive summary
5. **Email Management**: Summarize long email threads
6. **Social Media**: Content moderation summaries
7. **Research**: Abstract generation from papers

---

## 🔬 **Technical Deep Dive: Why This Matters for MLOps**

### **CI/CD Readiness**

| Aspect | Implementation | Production Benefit |
|--------|----------------|-------------------|
| **Testing** | Modular pipelines | Easy unit tests per stage |
| **Deployment** | Docker + FastAPI | Kubernetes-ready |
| **Monitoring** | Structured logging | Integrate with ELK/Splunk |
| **Versioning** | YAML configs | Git-trackable experiments |
| **Scalability** | Stateless API | Horizontal scaling |

---

### **The MLOps Maturity This Represents**

**Level 0** (Manual): Run notebooks, export model manually  
**Level 1** (Scripts): Python scripts for training  
**Level 2** (Pipelines): Automated pipeline (← **This Project**)  
**Level 3** (CI/CD): Automated testing & deployment  
**Level 4** (Production): Monitoring, A/B testing, auto-retraining  

**📌 This project is at Level 2**, with clear paths to Levels 3-4 (see Future Enhancements).

---

## 🎯 **Conclusions: What We Learn**

### **1. Production ML is Engineering-Heavy**
The model (PEGASUS) is ~10% of the work. The other 90%:
- Data pipelines
- API development
- Configuration management
- Evaluation frameworks
- Documentation
- Deployment infrastructure

### **2. Modularity Enables Iteration**
By separating stages, you can:
- Swap models (PEGASUS → BART → T5) without rewriting everything
- Add monitoring without touching training code
- Scale components independently

### **3. Transfer Learning is Powerful**
Fine-tuning pre-trained models:
- Requires 100x less data
- Trains 50x faster
- Achieves comparable results

### **4. Documentation = Professional Maturity**
This README demonstrates:
- Clear communication
- Anticipating user questions
- Lowering onboarding friction

---

## 💼 **Why This Matters for Your Portfolio**

**When a recruiter sees this project, they conclude:**

✅ You understand **full ML lifecycle** (not just training)  
✅ You write **production-grade code** (not just prototypes)  
✅ You know **modern tools** (Transformers, FastAPI, Docker)  
✅ You can **communicate clearly** (documentation)  
✅ You think like an **engineer**, not just a data scientist  

**Translation:** You're ready to contribute to **real ML teams** building **real products**.

---

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Google Research for the PEGASUS model
- SAMSum dataset creators
- FastAPI community for excellent documentation

---

## 📚 References

- [PEGASUS Paper](https://arxiv.org/abs/1912.08777) - Zhang et al., 2020
- [SAMSum Dataset](https://arxiv.org/abs/1911.12237) - Gliwa et al., 2019
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

<p align="center">
  <strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ and Python
</p>
