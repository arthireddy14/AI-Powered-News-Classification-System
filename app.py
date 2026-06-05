
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set sleek page configuration layout parameters
st.set_page_config(
    page_title="AI News Classifier", page_icon="📰", layout="wide"
)

# Wipe out internal cached memory profiles to guarantee smooth initialization execution runs
tf.keras.backend.clear_session()

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATIONS (Match your notebook settings)
# ---------------------------------------------------------
# Note: Ensure these parameters match what you used inside your training notebook loop execution run
MAX_WORDS = 10000  # Size constraint threshold cap set inside your tokenizer
MAX_LEN = 150  # Sentence ceiling width matrix alignment size


# ---------------------------------------------------------
# FUNCTIONAL DESIGN ARCHITECTURE RE-BUILDER
# ---------------------------------------------------------
def build_attention_architecture():
    """Rebuilds the layer network graph schema exactly to safely load weights array tensors."""
    attn_inputs = Input(shape=(MAX_LEN,), name="Input")
    attn_embedding = Embedding(input_dim=MAX_WORDS, output_dim=128, name="Embedding")(
        attn_inputs
    )

    # Core execution configuration layer returning extraction arrays
    mha_output, attention_weights = MultiHeadAttention(
        num_heads=4, key_dim=128, name="MultiHeadAttention"
    )(attn_embedding, attn_embedding, return_attention_scores=True)

    pooling = GlobalAveragePooling1D(name="GlobalAveragePooling1D")(mha_output)
    attn_outputs = Dense(5, activation="softmax", name="Output")(pooling)

    built_model = Model(inputs=attn_inputs, outputs=[attn_outputs, attention_weights])
    return built_model


# ---------------------------------------------------------
# DEPENDENCY ASYNC STORAGE RETRIEVAL LOADING PIPELINE
# ---------------------------------------------------------
@st.cache_resource
def load_application_artifacts():
    try:
        # 1. Load Preprocessing State
        if not os.path.exists("tokenizer (1).pkl") or not os.path.exists(
            "label_encoder.pkl"
        ):
            st.error(
                "❌ Missing files! Please verify 'tokenizer.pkl' and 'label_encoder.pkl' match app.py folder coordinates."
            )
            return None, None, None

        with open("tokenizer (1).pkl", "rb") as f:
            tokenizer = pickle.load(f)

        with open("label_encoder.pkl", "rb") as f:
            category_mapping = pickle.load(f)

        # 2. Build model architecture scaffolding
        model = build_attention_architecture()

        # 3. Load weight values matrix cleanly using by_name=True to resolve layer map structural bugs
        if os.path.exists("model (1).h5"):
            model.load_weights("model (1).h5", by_name=True, skip_mismatch=True)
        else:
            st.error(
                "❌ Missing structural binary graph layer parameters file 'model.h5'!"
            )
            return None, None, None

        return model, tokenizer, category_mapping

    except Exception as e:
        st.error(f"💥 Failed to initialize application graph pipeline. Error: {e}")
        return None, None, None


# Initialize components
model, tokenizer, category_mapping = load_application_artifacts()

if category_mapping:
    # Reverse lookups linking index targets back to structural display text headers cleanly
    reverse_mapping = {v: k.upper() for k, v in category_mapping.items()}


# ---------------------------------------------------------
# TEXT REGEX CLEANING METHOD
# ---------------------------------------------------------
def clean_input_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Strips special string characters
    return text


# ---------------------------------------------------------
# FRONT-END WEB USER INTERFACE STRUCTURE
# ---------------------------------------------------------
st.title("📰 AI-Powered News Classification System")
st.markdown(
    """
This production dashboard parses text blocks, categorizes primary contexts via 
**Multi-Head Attention Layers**, and highlights deep **word-to-word relationship tracking matrices** live.
"""
)
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input News Document Text")
    user_input = st.text_area(
        "Paste the raw article content body below:",
        height=300,
        placeholder="Example: Apple launches new AI-powered iPhone with state-of-the-art features...",
    )
    submit_button = st.button("Classify & Map Relationships", type="primary")

with col2:
    st.subheader("Model Classification Analysis")

    if submit_button and user_input.strip() != "":
        if model and tokenizer and category_mapping:
            with st.spinner("Analyzing matrix tokens and attention routing layers..."):
                # Clean text context parameters
                cleaned = clean_input_text(user_input)
                input_words = cleaned.split()

                if len(input_words) == 0:
                    st.warning("Provided text string evaluates to empty sequence space parameters.")
                else:
                    # Tokenize string array items into structured index arrays
                    sequence = tokenizer.texts_to_sequences([cleaned])
                    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

                    # Run structural neural graph prediction inferences
                    predictions, weights_matrix = model.predict(padded, verbose=0)

                    # Extract targets matching evaluation arrays
                    predicted_idx = np.argmax(predictions[0])
                    predicted_category = reverse_mapping[predicted_idx]
                    confidence_score = predictions[0][predicted_idx] * 100

                    # Render Metrics Block UI Card Layouts
                    st.metric(
                        label="Predicted Category Outcome",
                        value=predicted_category,
                        delta=f"{confidence_score:.2f}% Confidence Scale",
                    )

                    st.write("---")
                    st.subheader("🧠 Word-to-Word Attention Heatmap Matrix")
                    st.caption(
                        "X-axis matches Keys (where contextual attention flows); Y-axis matches Queries (evaluating token contexts)."
                    )

                    # Average values calculated across multihead channel dimensions
                    avg_weights = np.mean(weights_matrix[0], axis=0)

                    # Bound visual layout tracking sizes down to sentence limits (cap at 12 words max for clear display legibility)
                    display_limit = min(len(input_words), 12)
                    heatmap_data = avg_weights[:display_limit, :display_limit]
                    display_labels = input_words[:display_limit]

                    # Standardize normalization bounds to secure chart stability ratios
                    if np.sum(heatmap_data) > 0:
                        heatmap_data = heatmap_data / np.sum(
                            heatmap_data, axis=1, keepdims=True
                        )

                    # Render Seaborn Graph canvas element blocks
                    fig, ax = plt.subplots(figsize=(6.5, 5.5))
                    sns.heatmap(
                        heatmap_data,
                        annot=True,
                        fmt=".2f",
                        cmap="YlGnBu",
                        xticklabels=display_labels,
                        yticklabels=display_labels,
                        ax=ax,
                        cbar=False,
                    )
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
                    plt.tight_layout()

                    # Send generated figure vector map plot straight onto UI layout blocks
                    st.pyplot(fig)
        else:
            st.error("Missing underlying system graph files.")
    elif submit_button:
        st.warning("Please provide a valid text string inside the workspace window.")