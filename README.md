# 🎵 Spotify Music Recommendation System

A content-based music recommendation system built using unsupervised machine learning techniques on Spotify song data. This project combines clustering and cosine similarity to recommend songs based on audio features and engagement metrics without relying on user listening history.

### 📌 Project Overview

The rapid growth of music streaming platforms has created an information overload problem where users struggle to discover songs that match their preferences. This project addresses the issue by building a recommendation system that works even without historical user interaction data.

The system uses:

K-Means Clustering
Cosine Similarity
Feature Engineering
Content-Based Filtering
Streamlit Interactive Application

The recommendation engine groups songs with similar characteristics and generates ranked recommendations based on selected favourite songs.

### 🚀 Features
- Song recommendation based on audio similarity
- K-Means clustering for grouping similar songs
- Search songs by artist, track, or album
- Select favourite songs interactively
- Fast recommendation generation using memory-mapped similarity matrix
- Interactive Streamlit web application

### 🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- K-Means Clustering
- Cosine Similarity
- PCA

### 📂 Dataset
The Spotify dataset used in this project can be obtained from the following link:
https://app.gigasheet.com/spreadsheet/Spotify-dataset/b3f749fe_7428_4de0_a9fa_6577c94d2c57/4af52a2d_1287_4f45_b700_cc4e52c9176b

The dataset contains:

Audio features
- Popularity metrics
- Engagement statistics
- Song metadata

Including:
- Danceability
- Energy
- Loudness
- Tempo
- Streams
- Likes
- Comments
- Album information

### ⚙️ Project Workflow
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Feature Engineering
4. Feature Selection
5. K-Means Clustering
6. Similarity Matrix Construction
7. Recommendation System Development
8. Streamlit Application Deployment

### 📁 Project Structure
- app.py
- data_preparation.ipynb
- README.md

### ▶️ How to Run
1. Run Data Preparation First
Before launching the application, make sure to run:
```
data_preparation.ipynb
```
This notebook performs:

Data preprocessing
Feature engineering
Clustering
Similarity matrix generation

2. Run the Streamlit Application
After completing data preparation, run:
```
streamlit run app.py
```
### 📌 Key Results
- Optimal cluster count: k = 2
- Final feature matrix size: 20,512 × 15
- High recommendation consistency
- 100% cluster coherence in testing
- Efficient recommendation generation using precomputed similarity matrix

### Authors
- Muhammad Afiq bin Muhd Azri Fahmi
- Muhammad Hakeem Hadi bin Fairuz
- Anukthai Seeyakmani A/L Kuson
- Muhammad Kamil Zaki bin Iskandar Al-Thani
