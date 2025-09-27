import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import pandas as pd
import time
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="🧠 Neural Knowledge Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect theme and apply appropriate CSS
def apply_theme_css():
    # Auto-detect theme based on Streamlit's theme settings
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Light theme styles (default) */
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --text-primary: #1a1a1a;
            --text-secondary: #4a4a4a;
            --text-tertiary: #6b6b6b;
            --border-color: #e0e0e0;
            --accent-color: #2563eb;
            --accent-hover: #1d4ed8;
            --success-color: #16a34a;
            --error-color: #dc2626;
            --warning-color: #d97706;
            --card-shadow: rgba(0, 0, 0, 0.1);
        }
        
        /* Dark theme styles */
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #1e1e1e;
                --bg-secondary: #2a2a2a;
                --text-primary: #ffffff;
                --text-secondary: #e0e0e0;
                --text-tertiary: #b0b0b0;
                --border-color: #404040;
                --accent-color: #3b82f6;
                --accent-hover: #2563eb;
                --success-color: #22c55e;
                --error-color: #ef4444;
                --warning-color: #f59e0b;
                --card-shadow: rgba(0, 0, 0, 0.3);
            }
        }
        
        .main-header {
            color: var(--text-primary);
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px var(--card-shadow);
        }
        
        .subtitle {
            text-align: center;
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        .passage-card {
            background: var(--bg-primary);
            border: 2px solid var(--border-color);
            border-left: 5px solid var(--accent-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px var(--card-shadow);
            color: var(--text-primary);
        }
        
        .passage-card h3 {
            color: var(--accent-color);
            margin-bottom: 0.8rem;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .passage-card p {
            color: var(--text-secondary);
            line-height: 1.6;
            font-size: 0.95rem;
            margin: 0;
        }
        
        .similarity-score {
            background: var(--success-color);
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8rem;
            float: right;
            box-shadow: 0 2px 4px var(--card-shadow);
        }
        
        .source-tag {
            background: var(--warning-color);
            color: white;
            padding: 0.3rem 0.6rem;
            border-radius: 15px;
            font-size: 0.75rem;
            font-weight: 600;
            box-shadow: 0 2px 4px var(--card-shadow);
        }
        
        .upload-section {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 2rem;
            border: 3px dashed var(--accent-color);
            margin: 1rem 0;
            color: var(--text-primary);
        }
        
        .upload-section h3 {
            color: var(--accent-color);
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .upload-section p, .upload-section li {
            color: var(--text-secondary);
            line-height: 1.5;
        }
        
        .status-message {
            background: var(--accent-color);
            color: white;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 500;
            box-shadow: 0 2px 4px var(--card-shadow);
        }
        
        .error-message {
            background: var(--error-color);
            color: white;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 500;
        }
        
        /* Streamlit component overrides */
        .stButton > button {
            background: var(--accent-color) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background: var(--accent-hover) !important;
            transform: translateY(-1px) !important;
        }
        
        .stTextInput label, .stSelectbox label, .stTextArea label, .stFileUploader label {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }
        
        .stTextInput > div > div > input {
            background: var(--bg-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
        }
        
        .stSelectbox > div > div {
            background: var(--bg-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        
        .stTextArea > div > div > textarea {
            background: var(--bg-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        
        .stMetric {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }
        
        .stMetric label {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }
        
        .stMetric [data-testid="metric-value"] {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        
        .stSuccess {
            background: rgba(34, 197, 94, 0.1) !important;
            color: var(--success-color) !important;
            border: 1px solid var(--success-color) !important;
        }
        
        .stError {
            background: rgba(239, 68, 68, 0.1) !important;
            color: var(--error-color) !important;
            border: 1px solid var(--error-color) !important;
        }
        
        .stWarning {
            background: rgba(245, 158, 11, 0.1) !important;
            color: var(--warning-color) !important;
            border: 1px solid var(--warning-color) !important;
        }
        
        .stInfo {
            background: rgba(59, 130, 246, 0.1) !important;
            color: var(--accent-color) !important;
            border: 1px solid var(--accent-color) !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_theme_css()

# Initialize session state with error handling
def init_session_state():
    defaults = {
        'embeddings': None,
        'faiss_index': None,
        'model': None,
        'passages': None,
        'search_history': [],
        'current_source': "sample",
        'processing_errors': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Sample datasets (lightweight)
SAMPLE_DATA = {
    "🌌 Space": [
        {"title": "Mars Mission", "text": "Mars exploration continues with rover missions searching for signs of ancient life. Recent discoveries include evidence of past water activity and organic compounds in Martian soil samples.", "source": "sample"},
        {"title": "Space Station", "text": "The International Space Station serves as humanity's laboratory in space, conducting experiments in microgravity that advance our understanding of physics, biology, and materials science.", "source": "sample"},
        {"title": "Space Telescope", "text": "Advanced space telescopes like James Webb are revolutionizing astronomy by observing distant galaxies and exoplanets, providing unprecedented insights into cosmic evolution.", "source": "sample"},
    ],
    "🧬 Biology": [
        {"title": "Gene Editing", "text": "CRISPR technology enables precise DNA modifications, opening possibilities for treating genetic diseases and improving agricultural crops through targeted genetic changes.", "source": "sample"},
        {"title": "Vaccines", "text": "mRNA vaccine technology represents a breakthrough in immunization, allowing rapid development of vaccines against emerging diseases by instructing cells to produce protective proteins.", "source": "sample"},
        {"title": "Synthetic Biology", "text": "Scientists are engineering biological systems to create new materials, medicines, and sustainable fuels by designing custom organisms with specific functions.", "source": "sample"},
    ],
    "🤖 AI Tech": [
        {"title": "Machine Learning", "text": "Machine learning algorithms learn patterns from data to make predictions and decisions, powering applications from image recognition to language translation.", "source": "sample"},
        {"title": "Neural Networks", "text": "Deep neural networks process complex data through multiple layers, enabling breakthroughs in computer vision, natural language processing, and game playing.", "source": "sample"},
        {"title": "AI Safety", "text": "Ensuring AI systems remain beneficial and aligned with human values is crucial as artificial intelligence becomes more powerful and widespread.", "source": "sample"},
    ]
}

@st.cache_resource
def load_model():
    """Load sentence transformer model with error handling"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load AI model: {str(e)}")
        return None

def safe_chunk_text(text, max_length=400, overlap=50):
    """Safely chunk text with error handling"""
    try:
        if not text or len(text.strip()) < 50:
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        
        if len(words) < 10:
            return [text]
        
        chunks = []
        for i in range(0, len(words), max_length - overlap):
            chunk = ' '.join(words[i:i + max_length])
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
        
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text] if text else []

def safe_extract_pdf(pdf_file):
    """Safely extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        
        for page_num, page in enumerate(pdf_reader.pages[:10]):  # Limit to 10 pages
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"Page {page_num + 1}: {page_text}")
            except Exception as e:
                logger.warning(f"Error reading page {page_num + 1}: {e}")
                continue
        
        return '\n\n'.join(text_parts) if text_parts else None
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return None

def safe_scrape_webpage(url, timeout=10):
    """Safely scrape webpage with error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        return text[:10000] if text else None  # Limit to 10k characters
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return None

def process_content_to_passages(content, source_name, source_type):
    """Process content into searchable passages"""
    if not content:
        return []
    
    try:
        chunks = safe_chunk_text(content)
        passages = []
        
        for i, chunk in enumerate(chunks[:20]):  # Limit to 20 chunks
            if len(chunk.strip()) > 100:
                passages.append({
                    'title': f"{source_name} - Section {i+1}",
                    'text': chunk,
                    'source': source_type,
                    'source_detail': source_name
                })
        
        return passages
    except Exception as e:
        logger.error(f"Error processing content: {e}")
        return []

def create_embeddings_safely(passages, model):
    """Create embeddings with comprehensive error handling"""
    if not passages or not model:
        return None, None
    
    try:
        # Validate passages
        valid_passages = [p for p in passages if p.get('text') and len(p['text'].strip()) > 10]
        if not valid_passages:
            st.error("No valid passages to process")
            return None, None
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        embeddings = []
        
        for i, passage in enumerate(valid_passages):
            try:
                status_text.text(f"Processing {i+1}/{len(valid_passages)}: {passage['title'][:50]}...")
                text = f"{passage['title']}: {passage['text']}"
                embedding = model.encode([text])
                embeddings.append(embedding[0])
                progress_bar.progress((i + 1) / len(valid_passages))
            except Exception as e:
                logger.warning(f"Error encoding passage {i}: {e}")
                continue
        
        if not embeddings:
            st.error("Failed to create any embeddings")
            return None, None
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        # Clean up UI
        progress_bar.empty()
        status_text.empty()
        
        # Update passages to match successful embeddings
        successful_passages = valid_passages[:len(embeddings)]
        
        return embeddings_array, index, successful_passages
        
    except Exception as e:
        st.error(f"Embedding creation failed: {str(e)}")
        return None, None, None

def safe_search(query, model, index, passages, k=3):
    """Perform safe semantic search"""
    if not all([query, model, index, passages]):
        return []
    
    try:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        k = min(k, len(passages))
        scores, indices = index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if 0 <= idx < len(passages):
                results.append({
                    'passage': passages[idx],
                    'similarity': max(0.0, float(score)),  # Ensure non-negative
                    'rank': i + 1
                })
        
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def create_visualization(embeddings, passages, query_embedding=None):
    """Create safe visualization with error handling"""
    try:
        if embeddings is None or len(embeddings) < 2:
            return None
        
        # Ensure arrays match
        min_len = min(len(embeddings), len(passages))
        embeddings = embeddings[:min_len]
        passages = passages[:min_len]
        
        # PCA reduction
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create visualization data
        titles = [p.get('title', f'Passage {i}') for i, p in enumerate(passages)]
        sources = [p.get('source', 'unknown') for p in passages]
        
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hover_name=titles,
            color=sources,
            title="Knowledge Space Map",
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
        )
        
        # Add query point if provided
        if query_embedding is not None:
            try:
                query_2d = pca.transform(query_embedding.reshape(1, -1))
                fig.add_trace(go.Scatter(
                    x=query_2d[:, 0], y=query_2d[:, 1],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Your Query'
                ))
            except:
                pass  # Skip query point if transformation fails
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Neural Knowledge Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload PDFs, scrape websites, or explore sample data with AI search</p>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model()
    
    if not st.session_state.model:
        st.error("Failed to load AI model. Please refresh and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Data Source")
        
        source_type = st.selectbox(
            "Choose source:",
            ["📚 Sample Data", "📄 Upload PDF", "🌐 Web Scraping", "✍️ Custom Text"]
        )
        
        # Handle different sources
        if source_type == "📚 Sample Data":
            dataset = st.selectbox("Choose dataset:", list(SAMPLE_DATA.keys()))
            if st.button("Load Sample"):
                st.session_state.passages = SAMPLE_DATA[dataset].copy()
                st.session_state.current_source = "sample"
                st.session_state.embeddings = None  # Reset embeddings
                st.rerun()
        
        elif source_type == "📄 Upload PDF":
            uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'])
            if uploaded_file and st.button("Process PDF"):
                with st.spinner("Extracting PDF content..."):
                    content = safe_extract_pdf(uploaded_file)
                if content:
                    passages = process_content_to_passages(content, uploaded_file.name, "pdf")
                    if passages:
                        st.session_state.passages = passages
                        st.session_state.embeddings = None
                        st.success(f"Loaded {len(passages)} passages!")
                        st.rerun()
        
        elif source_type == "🌐 Web Scraping":
            url = st.text_input("Enter URL:", placeholder="https://example.com")
            if url and st.button("Scrape Website"):
                if url.startswith(('http://', 'https://')):
                    with st.spinner("Scraping website..."):
                        content = safe_scrape_webpage(url)
                    if content:
                        domain = urlparse(url).netloc
                        passages = process_content_to_passages(content, domain, "web")
                        if passages:
                            st.session_state.passages = passages
                            st.session_state.embeddings = None
                            st.success(f"Loaded {len(passages)} passages!")
                            st.rerun()
                else:
                    st.error("Please enter a valid URL")
        
        elif source_type == "✍️ Custom Text":
            text = st.text_area("Paste your text:", height=150)
            if text and st.button("Process Text"):
                passages = process_content_to_passages(text, "Custom Input", "custom")
                if passages:
                    st.session_state.passages = passages
                    st.session_state.embeddings = None
                    st.success(f"Loaded {len(passages)} passages!")
                    st.rerun()
        
        # Stats
        if st.session_state.passages:
            st.markdown("### 📈 Stats")
            st.metric("Passages", len(st.session_state.passages))
            if st.session_state.search_history:
                st.metric("Searches", len(st.session_state.search_history))
        
        if st.button("🗑️ Clear All"):
            for key in ['passages', 'embeddings', 'faiss_index', 'search_history']:
                st.session_state[key] = None if key != 'search_history' else []
            st.rerun()
    
    # Main content
    if not st.session_state.passages:
        st.markdown("""
        <div class="upload-section">
            <h3>🚀 Get Started</h3>
            <p>Choose a data source from the sidebar:</p>
            <ul>
                <li><strong>Sample Data:</strong> Try pre-loaded examples</li>
                <li><strong>Upload PDF:</strong> Extract and search PDF content</li>
                <li><strong>Web Scraping:</strong> Analyze website content</li>
                <li><strong>Custom Text:</strong> Paste your own content</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Create embeddings
    if st.session_state.embeddings is None:
        st.markdown('<div class="status-message">Creating knowledge embeddings...</div>', unsafe_allow_html=True)
        result = create_embeddings_safely(st.session_state.passages, st.session_state.model)
        
        if result and len(result) == 3:
            embeddings, index, updated_passages = result
            st.session_state.embeddings = embeddings
            st.session_state.faiss_index = index
            st.session_state.passages = updated_passages
            st.success(f"✅ Ready! {len(updated_passages)} passages indexed")
        else:
            st.error("Failed to create embeddings")
            return
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("🔍 Search your knowledge base:", placeholder="What would you like to know?")
    with col2:
        search_btn = st.button("🚀 Search", type="primary", use_container_width=True)
    
    # Perform search
    if (search_btn or query) and query.strip():
        results = safe_search(query, st.session_state.model, st.session_state.faiss_index, st.session_state.passages)
        
        if results:
            # Store search
            st.session_state.search_history.append({
                'query': query,
                'results': len(results),
                'timestamp': datetime.now()
            })
            
            # Display results
            st.markdown("## 🎯 Search Results")
            
            for result in results:
                passage = result['passage']
                similarity = result['similarity'] * 100
                
                # Source icon mapping
                icons = {'sample': '📚', 'pdf': '📄', 'web': '🌐', 'custom': '✍️'}
                icon = icons.get(passage.get('source', 'custom'), '📄')
                
                st.markdown(f"""
                <div class="passage-card">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                        <span class="source-tag">{icon} {passage.get('source_detail', 'Unknown')}</span>
                        <span class="similarity-score">{similarity:.0f}% match</span>
                    </div>
                    <h3>#{result['rank']}: {passage.get('title', 'Untitled')}</h3>
                    <p>{passage.get('text', 'No content')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization (optional)
            if len(st.session_state.passages) > 2:
                st.markdown("## 🗺️ Knowledge Map")
                try:
                    query_emb = st.session_state.model.encode([query])
                    fig = create_visualization(st.session_state.embeddings, st.session_state.passages, query_emb)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Visualization unavailable for this search")
        else:
            st.warning("No relevant passages found. Try different keywords.")
    
    elif st.session_state.passages:
        st.markdown("## 📚 Knowledge Base Ready")
        st.info(f"💡 {len(st.session_state.passages)} passages ready for search!")
        
        # Show preview
        with st.expander("👀 Preview Content"):
            for i, passage in enumerate(st.session_state.passages[:3]):
                st.markdown(f"**{passage.get('title', f'Passage {i+1}')}**")
                st.write(passage.get('text', 'No content')[:150] + "...")
                if i < 2:
                    st.markdown("---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main function error: {e}")