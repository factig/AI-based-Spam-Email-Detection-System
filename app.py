import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

# Load models and vectorizer
nb_model = joblib.load("nb_model.pkl")
lr_model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize LIME explainer
class_names = ['Not Spam', 'Spam']
explainer = LimeTextExplainer(class_names=class_names)

# Streamlit Page Configuration
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

# App Title and Description
st.title("📧 AI-based Spam Email Detection System with Explainability")
st.markdown("""
Welcome! Enter your email text or upload a `.txt` file below to check if it's spam or not.
Choose your preferred model and get detailed insights on the prediction along with explanation.
""")

# Input method selection
input_method = st.radio("How would you like to provide the email content?", 
                        options=["📝 Type Email", "📂 Upload .txt File"])

email_text = ""
if input_method == "📝 Type Email":
    email_text = st.text_area("Type your email content here:", height=150, placeholder="Paste your email text...")
else:
    uploaded_file = st.file_uploader("Upload your email text file (.txt)", type=["txt"])
    if uploaded_file:
        try:
            email_text = uploaded_file.read().decode("utf-8")
            st.success("File loaded successfully!")
            st.text_area("Email content preview:", value=email_text, height=150)
        except Exception:
            st.error("Error reading the file. Please upload a valid UTF-8 encoded .txt file.")

# Model selection

model_choice = st.selectbox(
    "Select Classifier Model",
    options=["Naive Bayes", "Logistic Regression"],
    help="Naive Bayes: Simple and fast. Logistic Regression: Interpretable and robust.",
    index=0,  # default option
    placeholder="Choose a model..."  # optional placeholder text
)


# Button state
is_button_disabled = not bool(email_text.strip())

# Classify button
button_clicked = st.button("🔍 Classify Email", disabled=is_button_disabled)

if button_clicked:
    with st.spinner("Analyzing the email..."):
        # Transform and classify
        X_input = vectorizer.transform([email_text])
        model = nb_model if model_choice == "Naive Bayes" else lr_model
        pred = model.predict(X_input)[0]
        proba = np.max(model.predict_proba(X_input))
        label = "🚫 Spam" if pred == 1 else "✅ Not Spam"

        # Display result
        color = "#d90429" if pred == 1 else "#2b9348"
        st.markdown(f"<h2 style='color:{color};'>{label}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score:** {proba:.2f}")

        # LIME explainability
        def predict_proba(texts):
            vect_texts = vectorizer.transform(texts)
            return model.predict_proba(vect_texts)

        explanation = explainer.explain_instance(email_text, predict_proba, num_features=10)

        # Custom styling for LIME output
        html_exp = explanation.as_html()
        custom_html = f"""
        <style>
            body, table, td, th, li, div {{
                background-color: #f5f5f5 !important;
                font-family: 'Segoe UI', sans-serif;
                color: black !important;
            }}
        </style>
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
            {html_exp}
        </div>
        """
        st.markdown("### 🔍 Explanation of Prediction (LIME):")
        st.components.v1.html(custom_html, height=450, scrolling=True)

        # Logistic Regression Feature Importance
        if model_choice == "Logistic Regression":
            coefs = model.coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            input_array = X_input.toarray()[0]
            top_indices = input_array.argsort()[::-1]
            top_features = [(feature_names[i], coefs[i]) for i in top_indices if input_array[i] > 0][:10]

            st.markdown("### 🔍 Top Influential Words (Logistic Regression):")
            for word, score in top_features:
                word_color = "#d90429" if score > 0 else "#2b9348"
                st.markdown(f"<span style='color:{word_color}; font-weight:bold;'>{word}</span> ({score:+.4f})", unsafe_allow_html=True)
        else:
            st.info("ℹ️ Feature importance visualization is not supported for Naive Bayes in this demo.")

# Footer
st.markdown("---")
st.markdown("<small style='color:gray;'>Built by Harsh</small>", unsafe_allow_html=True)
