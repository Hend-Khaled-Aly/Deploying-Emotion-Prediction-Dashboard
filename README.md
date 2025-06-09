# Deploying Emotion Prediction Dashboard

This repository contains the production-ready deployment of the [Emotion Prediction Dashboard](https://github.com/Hend-Khaled-Aly/Emotion-Prediction-Dashboard), an interactive web application built with Dash that analyzes social media usage and predicts users' dominant emotional states.

The dashboard is deployed live on Fly.io and can be accessed here:
🔗 [Live Demo](https://deployed-dashboard-sparkling-sunset-4809.fly.dev/)

---

## 🚀 Project Overview

The Emotion Prediction Dashboard was originally developed to explore the relationship between social media usage patterns and users’ emotional well-being. It uses a trained Random Forest model to predict emotions like happiness, sadness, anger, etc., based on metrics such as daily usage time, number of posts, likes, comments, and messages.

This repository focuses on deploying that application using:

🐳 Docker for containerization

☁️ Fly.io for hosting the web app

🐍 Dash for the interactive frontend

---

## 📂 Repository Structure

Deploying-Emotion-Prediction-Dashboard/

│

├── assets/                    # Images/icons for each predicted emotion

├── data/                      # Contains the dataset (social_media_emotions)

├── rf_pipeline.pkl            # Pre-trained Random Forest model pipeline

├── app.py                     # Main Dash application code

├── Dockerfile                 # Docker image setup

├── fly.toml                   # Fly.io configuration file

├── Procfile                  # For deployment process definition

├── requirements.txt           # Python dependencies

---

## 📦 Setup Instructions

If you'd like to run this project locally:
1. Clone the repository
  git clone https://github.com/your-username/Deploying-Emotion-Prediction-Dashboard.git
  cd Deploying-Emotion-Prediction-Dashboard

2. Build and run with Docker
  docker build -t emotion-dashboard .
  docker run -p 8050:8050 emotion-dashboard

3. Open your browser and visit:
  http://localhost:8050

---

## 🌐 Deployment

This app is deployed on Fly.io using:

- A Dockerfile for image creation
- A fly.toml configuration file
- Procfile to define the web process

to deploy:

fly launch
fly deploy

Note: You must have the Fly CLI installed and configured.

---

## 📎 Related Repository

This deployment builds upon the original dashboard and model development work. Visit the main project for data analysis, model training, and more:
🔗 Emotion Prediction Dashboard - [Development Repo](https://github.com/Hend-Khaled-Aly/Emotion-Prediction-Dashboard)

---

## 🛠️ Technologies Used

- Python
- Dash
- Pandas
- Scikit-learn
- Docker
- Fly.io

