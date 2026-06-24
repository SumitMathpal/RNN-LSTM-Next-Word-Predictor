import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config for premium look
st.set_page_config(
    page_title="RNN-LSTM Next Word Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphic design and premium styling
st.markdown(
    """
    <style>
    /* Dark theme overrides and global styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
        color: #f0f2f6;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Glassmorphism containers */
    div.stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #ffffff;
    }
    
    /* Input areas */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #00f2fe !important;
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.15) !important;
    }
    
    /* Sidebar glass effect */
    [data-testid="stSidebar"] {
        background-color: rgba(14, 17, 23, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Card container */
    .prediction-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        color: #00f2fe;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .prediction-text {
        font-size: 1.4rem;
        font-weight: 400;
        line-height: 1.6;
        color: #e2e8f0;
    }
    
    .predicted-word {
        color: #00f2fe;
        font-weight: 700;
        text-decoration: underline;
    }
    
    .highlight-word {
        background: linear-gradient(90deg, rgba(0,242,254,0.15) 0%, rgba(79,172,254,0.15) 100%);
        border-left: 3px solid #00f2fe;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Caching the resources for instant generation
@st.cache_resource
def load_assets():
    try:
        # Load Tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        # Load Max Sequence Length
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)
        
        # Load Keras LSTM Model
        model = tf.keras.models.load_model("lstm_model.keras")
        
        # Build reverse index
        reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        
        return tokenizer, max_len, model, reverse_word_index, None
    except Exception as e:
        return None, None, None, None, str(e)

tokenizer, max_len, model, reverse_word_index, load_error = load_assets()

# Sidebar configuration
st.sidebar.markdown("<h2 style='font-family: \"Space Grotesk\", sans-serif; color: #f0f2f6; font-size: 1.5rem; margin-top: 1rem;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if load_error:
    st.sidebar.error(f"❌ Error loading model assets: {load_error}")
else:
    st.sidebar.success("🔮 RNN-LSTM Model & Tokenizer loaded successfully!")

# App settings in Sidebar
num_words = st.sidebar.slider("Number of words to predict:", min_value=1, max_value=20, value=3, step=1)
temp = st.sidebar.slider("Sampling Temperature:", min_value=0.1, max_value=2.0, value=0.7, step=0.1,
                          help="Higher values make predicted text more creative and random; lower values make it more deterministic.")
predict_mode = st.sidebar.selectbox("Prediction Method:", ["Temperature-based Sampling", "Deterministic (Argmax)"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 📊 Model Architecture Info
    - **Model Type**: Sequential RNN
    - **Layers**: Embedding ➡️ LSTM (128 units) ➡️ Dense (10,000 output units)
    - **Vocab Size**: 8,978 tokens
    - **Max Sequence Length**: 745
    """
)

# App Title & Description
st.markdown("<h1>🔮 RNN-LSTM Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A premium real-time deep learning assistant for next word suggestion and text generation</p>", unsafe_allow_html=True)

# Sample input suggestions
suggestions = {
    "Select a pre-configured sample...": "",
    "Prompt 1: Once upon a time in a": "Once upon a time in a",
    "Prompt 2: Deep learning is a subset of": "Deep learning is a subset of",
    "Prompt 3: The data science project was": "The data science project was",
    "Prompt 4: Artificial intelligence has the power to": "Artificial intelligence has the power to",
    "Prompt 5: To be or not to be that is": "To be or not to be that is"
}

selected_suggestion = st.selectbox("💡 Quick-Start Prompts", list(suggestions.keys()))
default_input = suggestions[selected_suggestion]

# User text input
user_input = st.text_area("✍️ Input Text", value=default_input if default_input else "", height=120, placeholder="Type a sentence or prompt to begin generating predictions...")

def predict_next_word(input_text, temperature=1.0, deterministic=False):
    if not input_text.strip():
        return None, []
    
    # Tokenize and pad sequence
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    padded_token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')
    
    # Run model prediction
    predictions = model.predict(padded_token_list, verbose=0)[0]
    
    # Get top 5 choices before modifying probabilities for reporting
    top_5_indices = np.argsort(predictions)[-5:][::-1]
    top_5_probs = predictions[top_5_indices]
    
    # Convert logits to a formatted list of tuples (word, probability)
    top_5_choices = []
    for idx, prob in zip(top_5_indices, top_5_probs):
        word = reverse_word_index.get(idx, "<OOV>")
        top_5_choices.append((word, float(prob)))
        
    if deterministic or temperature <= 0.1:
        # Choose the word with absolute highest probability
        predicted_idx = np.argmax(predictions)
    else:
        # Temperature-scaled sampling
        # Apply temperature scaling to probabilities
        predictions = np.log(np.clip(predictions, 1e-10, 1.0)) / temperature
        exp_preds = np.exp(predictions)
        probabilities = exp_preds / np.sum(exp_preds)
        
        # Sample using scaled distribution
        predicted_idx = np.random.choice(len(probabilities), p=probabilities)
        
    predicted_word = reverse_word_index.get(predicted_idx, "")
    return predicted_word, top_5_choices

# Execution block
if user_input:
    if load_error:
        st.error("Cannot predict: Model assets failed to load.")
    else:
        # Run prediction
        with st.spinner("Analyzing text and running inference..."):
            current_text = user_input
            predicted_sequence = []
            immediate_next_top_5 = []
            
            # Predict sequentially
            for i in range(num_words):
                next_word, top_5 = predict_next_word(
                    current_text, 
                    temperature=temp, 
                    deterministic=(predict_mode == "Deterministic (Argmax)")
                )
                if not next_word:
                    break
                
                # Save the distribution details for the *very first* next word predicted
                if i == 0:
                    immediate_next_top_5 = top_5
                
                predicted_sequence.append(next_word)
                current_text += " " + next_word
            
            # Display results in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="card-title">✨ Generated Output</div>
                        <div class="prediction-text">
                            {user_input} <span class="highlight-word">{" ".join(predicted_sequence)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="card-title">📝 Summary of Predictions</div>
                        <p style="margin-bottom:0.5rem;"><strong>Immediate Next Word:</strong> <code style="font-size:1.1rem; color:#00f2fe;">{predicted_sequence[0] if predicted_sequence else ''}</code></p>
                        <p style="margin-bottom:0px;"><strong>Subsequent Words:</strong> <code>{', '.join(predicted_sequence[1:]) if len(predicted_sequence) > 1 else 'None'}</code></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with col2:
                if immediate_next_top_5:
                    st.markdown("<h3 style='font-family: \"Space Grotesk\", sans-serif; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 2px; color: #4facfe; margin-top: 1rem;'>📊 Next-Word Candidate Distribution</h3>", unsafe_allow_html=True)
                    
                    df_chart = pd.DataFrame(immediate_next_top_5, columns=["Word", "Confidence"])
                    df_chart = df_chart.sort_values(by="Confidence", ascending=True)
                    
                    fig = px.bar(
                        df_chart, 
                        x="Confidence", 
                        y="Word", 
                        orientation="h",
                        color="Confidence",
                        color_continuous_scale=["#0052d4", "#4facfe", "#00f2fe"],
                        labels={"Confidence": "Probability Score", "Word": "Candidate Word"},
                        height=260
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f0f2f6',
                        margin=dict(l=0, r=0, t=10, b=10),
                        coloraxis_showscale=False,
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also print table
                    st.markdown("<p style='font-size:0.85rem; color:#8892b0; margin-bottom: 0.5rem;'>Interactive Candidate Probability breakdown:</p>", unsafe_allow_html=True)
                    st.dataframe(
                        df_chart.sort_values(by="Confidence", ascending=False).style.format({"Confidence": "{:.2%}"}),
                        hide_index=True,
                        use_container_width=True
                    )
else:
    # Instructions if empty input
    st.info("💡 Enter some text or select one of the Quick-Start Prompts to begin forecasting candidate words.")
