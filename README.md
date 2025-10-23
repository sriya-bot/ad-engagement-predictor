# 📊 Ad Engagement Prediction Dashboard

An interactive **Streamlit web app** that predicts digital ad engagement levels using a trained **Random Forest Classifier** model.  
Built for data-driven marketers, this tool estimates campaign engagement confidence and provides AI-powered optimization insights.

---

## 🚀 Live Demo

👉 [**Open App**](https://ad-engagement-predictor-jewggrgxfdhlbkaqrqkhos.streamlit.app/)  

*(Runs directly in your browser — no installation required.)*

---

## 🧠 Project Overview

This project demonstrates end-to-end **Machine Learning model deployment** using Streamlit Cloud.  
The model analyzes advertising campaign parameters such as CTR, CPC, and keyword length to predict whether an ad will achieve **High** or **Low Engagement**.

---

## ⚙️ Features

✅ Real-time prediction of ad engagement level  
✅ Smart AI reasoning with actionable insights  
✅ Clean and professional Streamlit UI  
✅ Adjustable parameters for CTR, CPC, Leads, Device, and more  
✅ Ready for resume / portfolio showcase  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / Dashboard** | Streamlit |
| **Model** | Random Forest Classifier |
| **Language** | Python (NumPy, Pandas, Joblib, Scikit-learn) |
| **Deployment** | Streamlit Cloud |
| **Data** | Simulated Ad Campaign Data (CTR, CPC, Leads, etc.) |

---

## 📸 Output Screenshots

| Prediction | Insights |
|-------------|-----------|
| ![High Engagement](screenshots/high_engagement.png) | ![AI Insights](screenshots/insights.png) |

---

## 🧮 How It Works

1. **User Input:**  
   The user sets campaign metrics like CTR, CPC, and keyword length in the sidebar.

2. **Model Prediction:**  
   The Random Forest model (stored in `model.pkl`) predicts engagement probability.

3. **AI Insights:**  
   The app provides reasoning and practical suggestions to improve engagement scores.

4. **Visualization / Summary:**  
   Displays clear feedback such as:
   - 🔥 *High Engagement* (80%+ confidence)  
   - 🧭 *Near-High Engagement* (70–79%)  
   - ⚖️ *Moderate* (50–69%)  
   - 💤 *Low Engagement* (<50%)

---

## 📁 Repository Structure
📂 ad-engagement-predictor
┣ 📜 app.py
┣ 📜 train_model.py
┣ 📜 model.pkl
┣ 📜 model_columns.pkl
┣ 📜 requirements.txt
┣ 📜 README.md
┗ 📂 screenshots/
┣ 📸 high_engagement.png
┗ 📸 insights.png
