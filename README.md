# Meeting-Engagement-Analyzer
Meeting Engagement Analyzer is a machine learning-based system that analyzes participant behavior in online meetings to classify engagement levels, detect patterns, and provide actionable insights using both supervised and unsupervised learning techniques.

## 📊 Meeting Engagement Analyzer

This project is an intelligent system designed to analyze participant engagement in online meetings (e.g., Zoom or Microsoft Teams) using machine learning techniques.

The system processes meeting data such as attendance duration, chat activity, microphone usage, and screen sharing behavior to evaluate how actively participants are involved.

---

## 🎯 Objectives

* Classify participants into **High, Medium, or Low engagement levels**
* Detect behavioral patterns among attendees
* Identify distracted or passive participants
* Provide visual insights into overall meeting engagement

---

## ⚙️ Key Features

* 📂 Upload meeting data (CSV / Excel)
* 📈 Automated feature engineering
* 🤖 Multi-model machine learning predictions
* 👥 Clustering participants based on behavior
* 📊 Visualization using charts
* 🖥️ Interactive GUI interface

---

## 🧠 Machine Learning Models Used

### 🔹 Supervised Learning:

* **Random Forest** → Main model for engagement classification
* **Support Vector Machine (SVM)** → Binary engagement classification
* **K-Nearest Neighbors (KNN)** → Finding similar participants
* **Naive Bayes** → Fast probabilistic predictions

### 🔹 Unsupervised Learning:

* **K-Means Clustering** → Grouping participants into:

  * Active
  * Passive
  * Distracted

---

## 🧪 Feature Engineering

Custom features were created to improve model performance, such as:

* **Duration Ratio** → Attendance time relative to meeting duration
* **Participation Score** → Combined activity (chat, mic, screen share)
* **Late Join Indicator** → Whether a participant joined late

---

## 🏗️ Project Structure

```
project/
│── main.py
│── gui/
│── prediction/
│── training/
│── preprocessing/
│── clustering/
│── models/
│── data/
```

---

## 🚀 How It Works

1. User uploads meeting data
2. Data is preprocessed and scaled
3. Features are engineered
4. Multiple ML models generate predictions
5. Results are visualized and displayed

---

## 📊 Output

* Engagement Level (High / Medium / Low)
* Binary Engagement (Engaged / Not Engaged)
* Cluster Group (Active / Passive / Distracted)

---

## 💡 Motivation

With the rise of online meetings and remote learning, understanding participant engagement has become essential. This project aims to provide a data-driven approach to evaluate and improve interaction in virtual environments.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas & NumPy
* Matplotlib
* Tkinter (GUI)


لو عايزة version:

* 🔥 أقصر (for CV)
* 🔥 أو README شكله premium بـ badges و icons
* 🔥 أو GitHub bio/project pitch

قوليلي وأنا أظبطهولك 💯
