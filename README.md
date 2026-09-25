# Project 3 Capstone: Tech Stack Recommender
Built for **DecodeLabs Artificial Intelligence Industrial Training Kit (2026)**.

## 📌 Project Overview
This project is a content-based recommendation engine that maps a user's skills to relevant job roles (career paths). It uses **TF-IDF Weighting** and **Cosine Similarity** built entirely from scratch in pure Python without external ML libraries.

### Core Architecture
- **TF-IDF:** Down-weights common skills (like "Python") and boosts rare, descriptive skills (like "Kubernetes") to ensure accurate recommendations.
- **Cosine Similarity:** Measures the angle between your profile vector and job vectors to capture alignment regardless of length.

---

## 📂 Project Structure
```text
├── tech_stack_recommender.py  # Core Application Engine
├── raw_skills.csv             # Ingestion Database File
└── README.md                  # Project Documentation
```

---

## 🛠️ Setup & Dataset
1. Ensure **Python 3.x** is installed on your computer.
2. In the same folder as your script, ensure a file named `raw_skills.csv` exists with your dataset:
   ```csv
   job_role,skills
   Data Scientist,Python,SQL,Machine Learning,Data Analysis
   Cloud Engineer,Cloud Computing,AWS,Linux,Automation,Docker
   DevOps Engineer,Automation,Jenkins,Docker,Kubernetes,Linux
   Frontend Developer,HTML,CSS,JavaScript,React,Web Design
   ```

---

## 🚀 How to Run

### 1. Interactive Mode
Run the script normally and type your skills when prompted:
```bash
python tech_stack_recommender.py
```
*(Note: You must input a minimum of 3 skills separated by commas, e.g., `Python, SQL, Automation`)*

### 2. Command Line Arguments
Pass your skills directly when triggering the script:
```bash
python tech_stack_recommender.py "Python, Cloud Computing, Automation"
```
