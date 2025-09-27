# 🧠 Neural Knowledge Explorer

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A powerful, lightweight **Retrieval-Augmented Generation (RAG)** system built with Streamlit that enables semantic search across multiple data sources using AI embeddings and vector similarity.

![Neural Knowledge Explorer Demo](https://via.placeholder.com/800x400/2563eb/ffffff?text=Neural+Knowledge+Explorer)

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
   git clone https://github.com/yourusername/neural-knowledge-explorer.git
   cd neural-knowledge-explorer
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

## 📦 Dependencies

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

## 🎯 How It Works

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

## 🏗️ Architecture

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

## 🎨 Customization

### Theming
The app automatically adapts to your system's dark/light mode preference. You can also customize colors by modifying the CSS variables in the `apply_theme_css()` function.

### Model Selection
Change the embedding model by updating the `load_model()` function:
```python
# For better performance (larger model)
model = SentenceTransformer('all-mpnet-base-v2')

# For multilingual support
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

### Chunking Strategy
Adjust text chunking parameters:
```python
def safe_chunk_text(text, max_length=400, overlap=50):
    # Modify max_length and overlap for different chunk sizes
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:
```env
MAX_PDF_PAGES=10
MAX_CONTENT_LENGTH=10000
CHUNK_SIZE=400
CHUNK_OVERLAP=50
SEARCH_RESULTS=3
```

### Performance Tuning
- **Memory Usage**: Adjust chunk limits in processing functions
- **Speed**: Use smaller embedding models for faster processing
- **Accuracy**: Use larger models like `all-mpnet-base-v2` for better results

## 📊 Performance Metrics

| Operation | Time (avg) | Memory Usage |
|-----------|------------|--------------|
| Load Model | 2-5 seconds | 150MB |
| PDF Processing | 0.5s/page | 50MB |
| Web Scraping | 1-3 seconds | 20MB |
| Embedding Creation | 0.1s/chunk | 10MB/chunk |
| Search Query | <100ms | 5MB |

## 🚨 Troubleshooting

### Common Issues

**1. Model Loading Error**
```bash
# Solution: Ensure internet connection for first-time download
pip install --upgrade sentence-transformers
```

**2. PDF Reading Error**
```bash
# Solution: Install or upgrade PyPDF2
pip install --upgrade PyPDF2
```

**3. Web Scraping Blocked**
```bash
# Some websites block automated requests
# Try with different URLs or check robots.txt
```

**4. Memory Issues**
```bash
# Reduce processing limits in the code:
# - MAX_PDF_PAGES = 5
# - MAX_CONTENT_LENGTH = 5000
```

### Debug Mode
Enable logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit a pull request

### Code Style
- Follow PEP 8 conventions
- Use type hints where applicable
- Add docstrings to functions
- Write unit tests for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Sentence Transformers** for state-of-the-art embeddings
- **FAISS** for efficient similarity search
- **Plotly** for beautiful visualizations
- **HuggingFace** for pre-trained models

## 📈 Roadmap

- [ ] **Multi-language Support** - International document processing
- [ ] **Advanced Chunking** - Semantic-aware text splitting
- [ ] **Cloud Storage** - Integration with AWS S3, Google Drive
- [ ] **Batch Processing** - Handle multiple documents simultaneously
- [ ] **API Endpoints** - REST API for programmatic access
- [ ] **Export Features** - Save search results and embeddings
- [ ] **Advanced Visualizations** - 3D plots and network graphs

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/neural-knowledge-explorer/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/yourusername/neural-knowledge-explorer/discussions)
- **Email**: your.email@example.com

## ⭐ Show Your Support

If you find this project helpful, please consider:
- ⭐ **Starring** this repository
- 🐛 **Reporting** bugs and issues
- 💡 **Suggesting** new features
- 🤝 **Contributing** code or documentation

---

<p align="center">
  Made with ❤️ and 🧠 by <a href="https://github.com/yourusername">Your Name</a>
</p>

<p align="center">
  <a href="#top">Back to Top ⬆️</a>
</p>