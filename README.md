---
title: khojbin
sdk: docker
emoji: 🚀
colorFrom: blue
colorTo: red
license: mit
---
# 🎯 Intelligent Quiz Solver

An advanced, fully-automated quiz solver that handles multiple types of programming quizzes without any hardcoding. **100% dynamic and data-driven!**

## 🚀 Features

- ✅ **API Calls with Authentication**: Automatically detects and adds email/secret parameters
- ✅ **JavaScript Evaluation**: Executes complex JS code using LLM interpretation  
- ✅ **CSV Data Analysis**: Smart column selection and filtering based on question context
- ✅ **Image Analysis**: OCR text extraction from images
- ✅ **Audio Transcription**: Converts audio to text and extracts information
- ✅ **Mathematical Calculations**: Median, average, sum, count operations
- ✅ **Geographic Distances**: Haversine distance calculations
- ✅ **Data Visualization**: Chart generation and analysis
- ✅ **Machine Learning**: Linear regression predictions

## 🎯 Key Innovations

### No Hardcoding!
- All logic is data-driven and flexible
- Works with any cutoff values, column names, or data structures
- Semantic matching for column selection
- Context-aware authentication detection
- Automatic filter detection from question text

### Multi-Strategy Approach
**13+ Candidate Generation Strategies** for audio/cutoff quizzes:
- Greater than (>), Greater than or equal (>=)
- Less than (<), Less than or equal (<=)
- Equal to (==), Not equal to (!=)
- Even numbers, Odd numbers
- Combinations (>+even, >+odd, <=+even, <=+odd)

### Smart CSV Processing
- **Semantic Column Selection**: Matches column names to question keywords
  - "median profit" → selects 'profit' column
  - "average sales" → selects 'sales' column
- **Automatic Filtering**:
  - Year filters: "sold in 2024" → `df['year'] == 2024`
  - Comparison filters: "sales greater than 10000" → `df['sales'] > 10000`

### API Authentication
Detects auth requirements from question context:
- "include your email and secret"
- "must use authentication"
- "query parameters"
→ Automatically adds `?email=...&secret=...`

---

## 📦 Deployment to Hugging Face Spaces

### Quick Deploy

1. **Create a new Space** at https://huggingface.co/new-space
   - Name: `quiz-solver`
   - SDK: **Docker**
   - Hardware: CPU basic (free tier)

2. **Upload files** via web interface or Git:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/quiz-solver
   cd quiz-solver
   
   # Copy all files
   cp /path/to/main.py .
   cp /path/to/app.py .
   cp /path/to/Dockerfile .
   cp /path/to/requirements.txt .
   
   git add .
   git commit -m "Initial deployment"
   git push
   ```

3. **Set environment variables** in Space Settings:
   ```
   EMAIL=your.email@example.com
   SECRET=your_secret_key
   AIPIPE_TOKEN=your_aipipe_token
   LLM_MODEL=openai/o3-mini
   SUBMISSION_ATTEMPTS=15
   ```

4. **Wait for build** (~5-10 minutes for first build)

5. **Access your Space** at `https://huggingface.co/spaces/YOUR_USERNAME/quiz-solver`

---

## 🛠️ Local Setup

### Option 1: Using Docker

```bash
# Build the image
docker build -t quiz-solver .

# Run with environment file
docker run -p 7860:7860 --env-file .env quiz-solver
```

Access at: http://localhost:7860

### Option 2: Using Python

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run Gradio interface
python app.py
```

---

## 🔧 Configuration

Create a `.env` file:

```env
# Authentication
EMAIL=24f2002843@ds.study.iitm.ac.in
SECRET=abc123xyz

# LLM API
AIPIPE_TOKEN=your_aipipe_token_here
LLM_MODEL=openai/o3-mini

# Solver Settings
SUBMISSION_ATTEMPTS=15
```

### Get AIPipe Token

1. Visit https://aipipe.org/login
2. Sign in with Google
3. Copy token from dashboard
4. Paste in `.env` file

---

## 📊 How It Works

```
┌─────────────────────────────────────────────────────┐
│                 User Input (Quiz URL)               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            1. Selenium Web Scraper                  │
│  • Loads page with Chrome (handles JavaScript)     │
│  • Extracts question text, links, media files      │
│  • Downloads CSVs, PDFs, images, audio             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         2. Multi-Stage Question Analyzer            │
│  Stage 1: Check for explicit answers               │
│  Stage 2: Detect cutoff values                     │
│  Stage 2.5: Fetch API data (with auth detection)   │
│  Stage 2.7: Data cleaning tasks                    │
│  Stage 2.8: CSV filtering detection                │
│  Stage 3: Search for secret codes                  │
│  Stage 4: DataFrame calculations                   │
│  Stage 5: Visualization requests                   │
│  Stage 6: Image analysis                           │
│  Stage 7: Machine learning tasks                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            3. Smart Data Processing                 │
│  • Semantic column selection                       │
│  • Automatic filtering (year, comparisons)         │
│  • Multiple calculation strategies                 │
│  • 13+ candidate generation for audio quizzes      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           4. LLM Fallback (if needed)               │
│  • Uses appropriate model (o3-mini/gpt-4o/gemini)  │
│  • Sends question + all scraped data               │
│  • Extracts answer from JSON response              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│        5. Answer Submission & Validation            │
│  • Tries all candidates (up to 15 attempts)        │
│  • Checks server response                          │
│  • Follows chain to next quiz if available         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
                  Success!
```

---

## 🧪 Testing

### Test Single Quiz

```bash
curl -X POST http://localhost:7860/api/solve_quiz \
  -H "Content-Type: application/json" \
  -d '{
    "quiz_url": "https://p2testingone.vercel.app/q1.html",
    "email": "your.email@example.com",
    "secret": "your_secret"
  }'
```

### Test via Gradio UI

1. Open http://localhost:7860
2. Enter quiz URL
3. (Optional) Add email/secret
4. Click "🚀 Solve Quiz"
5. View detailed results

---

## 📁 Project Structure

```
quiz-solver/
├── app.py                 # Gradio web interface
├── main.py               # Core quiz solver logic
├── Dockerfile            # Container setup with Chrome
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md            # This file
└── downloads/           # Downloaded files (created automatically)
```

---

## 🔍 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Gradio 4.44.0 |
| **Backend** | FastAPI, Python 3.11 |
| **Browser** | Selenium + Chrome/ChromeDriver |
| **Data** | Pandas, NumPy |
| **LLM** | OpenAI API via Aipipe |
| **Container** | Docker |
| **Audio** | Whisper, ffmpeg |
| **Images** | Pillow |

---

## 🐛 Troubleshooting

### ChromeDriver Issues
```bash
pip install --upgrade webdriver-manager
```

### Memory Issues on Hugging Face
- Use CPU basic (free tier)
- Files auto-deleted after processing
- Streaming for large data

### LLM API Errors
- Check AIPIPE_TOKEN is valid
- Verify model name: `openai/o3-mini`
- Check API budget/limits

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🙏 Credits

Built with ❤️ as a flexible, data-driven quiz solver.

**Key Principles:**
- ✅ No hardcoding
- ✅ Data-driven logic
- ✅ Flexible strategies
- ✅ Automatic adaptation