"""
Streamlit Frontend for Sentiment Analysis API
Interactive web interface for real-time sentiment classification
"""

import streamlit as st
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# Styling
# ============================================================================

st.markdown("""
<style>
    .sentiment-badge-positive {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .sentiment-badge-negative {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .sentiment-badge-neutral {
        background-color: #6c757d;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .api-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    
    .confidence-text {
        font-size: 16px;
        font-weight: bold;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Constants
# ============================================================================

API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/"

SENTIMENT_EMOJIS = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐"
}

SENTIMENT_COLORS = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray"
}

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_health():
    """Check if API is running and healthy"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def predict_sentiment(text):
    """
    Send text to API for sentiment prediction
    
    Args:
        text: Input text string
    
    Returns:
        dict: Response with sentiment, confidence, label_id or error info
    """
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json={"text": text},
            timeout=10
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return {"success": False, "error": error_detail}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Is the API server running?"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Cannot connect to API at {API_URL}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_badge_html(sentiment):
    """Get HTML badge for sentiment display"""
    emoji = SENTIMENT_EMOJIS.get(sentiment, "❓")
    sentiment_display = sentiment.upper()
    
    if sentiment == "positive":
        return f'<div class="sentiment-badge-positive">{emoji} {sentiment_display}</div>'
    elif sentiment == "negative":
        return f'<div class="sentiment-badge-negative">{emoji} {sentiment_display}</div>'
    else:
        return f'<div class="sentiment-badge-neutral">{emoji} {sentiment_display}</div>'


# ============================================================================
# UI Components
# ============================================================================

# Header
st.markdown("# 💭 Sentiment Analyzer")
st.markdown("""
Analyze the emotional tone of your text in real-time.
Simply enter or paste your text below and click **Analyze Sentiment** to get started.
""")

st.divider()

# ============================================================================
# Main Interface
# ============================================================================

# Input section
st.subheader("📝 Your Text")
user_text = st.text_area(
    label="Enter or paste your text here:",
    placeholder="e.g., 'I absolutely love this product! It's amazing!'",
    height=150,
    label_visibility="collapsed"
)

# Analyze button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True,
        type="primary"
    )

st.divider()

# ============================================================================
# Results Section
# ============================================================================

if analyze_button:
    # Validate input
    if not user_text or not user_text.strip():
        st.error("⚠️ Please enter some text to analyze")
    else:
        # Check API health
        if not check_api_health():
            st.error(
                "❌ **API Server Unavailable**\n\n"
                f"Cannot connect to sentiment analysis API at `{API_URL}`\n\n"
                "**To run the API:**\n"
                "1. Open a terminal in the project directory\n"
                "2. Run: `python api.py`\n"
                "3. Wait for the message: '✓ API ready to serve requests'\n"
                "4. Then refresh this page\n\n"
                "**Required model files:**\n"
                "- `best_model_svm_glove_*.pkl` (GloVe) OR\n"
                "- `best_model_svm_tfidf_*.pkl` (TF-IDF)"
            )
        else:
            # Show loading state
            with st.spinner("🤔 Analyzing sentiment..."):
                result = predict_sentiment(user_text)
            
            # Process result
            if result["success"]:
                data = result["data"]
                sentiment = data["sentiment"]
                confidence = data["confidence"]
                
                # Display sentiment badge
                st.markdown(get_badge_html(sentiment), unsafe_allow_html=True)
                
                # Display confidence score
                st.markdown("### 📊 Confidence Score")
                
                # Progress bar
                st.progress(confidence, text=f"{confidence*100:.1f}%")
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Confidence",
                        value=f"{confidence*100:.1f}%",
                        delta=None
                    )
                with col2:
                    st.metric(
                        label="Class",
                        value=data["label_id"],
                        delta=None
                    )
                with col3:
                    st.metric(
                        label="Status",
                        value="✓ Success",
                        delta=None
                    )
                
                # Text summary
                st.markdown("### 📋 Analysis Summary")
                summary_text = f"""
                **Input Text:** {user_text[:100]}{'...' if len(user_text) > 100 else ''}
                
                **Predicted Sentiment:** {sentiment.capitalize()}
                
                **Confidence Level:** {confidence*100:.1f}%
                
                **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                st.info(summary_text)
                
                logger.info(f"✓ Prediction successful: {sentiment} ({confidence:.2%})")
            
            else:
                # Error handling
                error_msg = result["error"]
                
                st.error(
                    f"❌ **Analysis Failed**\n\n"
                    f"**Error:** {error_msg}\n\n"
                )
                
                # Provide helpful guidance
                if "Cannot connect" in error_msg or "timed out" in error_msg:
                    st.warning(
                        "**Troubleshooting:**\n"
                        "1. Is the API server running? (Check the terminal)\n"
                        "2. Try restarting the API: `python api.py`\n"
                        "3. Make sure a model file exists:\n"
                        "   - `best_model_svm_glove_*.pkl` (GloVe) OR\n"
                        "   - `best_model_svm_tfidf_*.pkl` (TF-IDF)"
                    )
                elif "empty" in error_msg.lower():
                    st.info("Please enter some text to analyze")
                
                logger.error(f"✗ Prediction failed: {error_msg}")

# ============================================================================
# Sidebar Info
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Sentiment Analyzer** powered by:
    - 🤖 SVM (Support Vector Machine)
    - 📝 Text Representation: GloVe 100D or TF-IDF
    - 🎯 Multi-class Classification (3 classes)
    
    **Supported Models:**
    - GloVe 100-dimensional embeddings
    - TF-IDF (1000 features)
    
    **Classes:**
    - Positive 😊
    - Neutral 😐
    - Negative 😞
    
    *The API automatically loads the latest model available*
    """)
    
    st.divider()
    
    st.markdown("### 🔧 API Status")
    
    api_health = check_api_health()
    if api_health:
        st.success("✓ API is running")
    else:
        st.error("✗ API is offline")
        st.markdown(
            f"**Endpoint:** `{API_URL}`\n\n"
            "Start the API with: `python api.py`"
        )
    
    st.divider()
    
    st.markdown("### 📚 Quick Examples")
    st.markdown("""
    **Positive:**
    - "I love this! Absolutely brilliant!"
    - "Best experience ever!"
    
    **Negative:**
    - "This is terrible and frustrating"
    - "Worst product ever!"
    
    **Neutral:**
    - "The weather is cloudy today"
    - "It's okay, nothing special"
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Sentiment Analysis API • Built with FastAPI & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
