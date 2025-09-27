# 🧠 Neural Knowledge Explorer

A powerful, lightweight **Retrieval-Augmented Generation (RAG)** system built with Streamlit that enables semantic search across multiple data sources using AI embeddings and vector similarity.

## ✨ Features

### 📚 **Multi-Source Data Support**
- **Sample Datasets** - Pre-loaded topics (Space, Biology, AI)
- **PDF Upload** - Extract and search PDF documents
- **Web Scraping** - Analyze website content in real-time
- **Custom Text** - Paste and search your own content

### 🧠 **Advanced AI Capabilities**
- **Semantic Search** - Understanding meaning, not just keywords
- **Vector Embeddings** - Using `sentence-transformers` for deep text understanding
- **FAISS Integration** - Lightning-fast similarity search
- **Smart Chunking** - Intelligent text segmentation with overlap

### 🎨 **Beautiful User Experience**
- **Adaptive Themes** - Auto dark/light mode support
- **Interactive Visualizations** - 2D knowledge space mapping with Plotly
- **Real-time Processing** - Live status updates and progress tracking
- **Responsive Design** - Works on desktop, tablet, and mobile

### 🛡️ **Robust & Reliable**
- **Comprehensive Error Handling** - Graceful failure recovery
- **Memory Efficient** - Smart limits and data validation
- **Safe Processing** - Input sanitization and bounds checking
- **Performance Optimized** - Cached models and efficient algorithms

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aliya-singh/virallens.git
   cd task4
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Dependencies

```txt
streamlit>=1.28.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
plotly>=5.15.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
PyPDF2>=3.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

## How It Works

### 1. **Data Ingestion**
The app supports multiple input methods:
- Upload PDF files for automatic text extraction
- Enter URLs for real-time web scraping
- Paste custom text content
- Use built-in sample datasets

### 2. **Text Processing**
- **Smart Chunking**: Text is split into overlapping segments (400 words with 50-word overlap)
- **Cleaning**: Removes unwanted elements from web content
- **Validation**: Ensures content quality before processing

### 3. **Embedding Generation**
- Uses `all-MiniLM-L6-v2` model for creating 384-dimensional embeddings
- Implements cosine similarity for semantic matching
- Stores vectors in FAISS index for efficient retrieval

### 4. **Semantic Search**
- Converts user queries into embeddings
- Performs similarity search against knowledge base
- Returns ranked results with confidence scores

### 5. **Visualization**
- PCA dimensionality reduction for 2D mapping
- Interactive scatter plots showing document relationships
- Query positioning in knowledge space

## 💡 Usage Examples

### Sample Dataset Search
```python
# Select "Sample Data" → "🌌 Space" → "Load Sample"
# Query: "Mars exploration missions"
# Returns: Relevant passages about Mars rovers and discoveries
```

### PDF Document Analysis
```python
# Upload research paper PDF
# Query: "machine learning applications"
# Returns: Relevant sections from the document with similarity scores
```

### Web Content Search
```python
# Enter URL: "https://en.wikipedia.org/wiki/Artificial_intelligence"
# Query: "neural networks"
# Returns: Extracted and chunked content matching the query
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Text Processing │───▶│   Embeddings    │
│  • PDF Files    │    │  • Chunking      │    │  • Sentence     │
│  • Web Pages    │    │  • Cleaning      │    │    Transformers │
│  • Custom Text  │    │  • Validation    │    │  • Vector Space │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Search Results │◀───│  Similarity      │◀───│  FAISS Index    │
│  • Ranked List  │    │  • Cosine Sim    │    │  • Fast Search  │
│  • Confidence   │    │  • Top-K Results │    │  • Normalized   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📸 Screenshots

| Source | Sample Data |
|-------------|--------------|
| ![Image1](sampledata1.jpeg) | ![Image2](sampledata1.jpeg) |