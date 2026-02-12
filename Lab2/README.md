# Lab 2 – Machine Learning Problem Identification & Methodology Design

**Course:** ARTI 308 – Machine Learning  
**Semester:** 2nd Semester 2025/2026  

---

## Dataset

**Name:** Netflix Movies and TV Shows  
**Source:** Kaggle  

The dataset contains metadata about Netflix titles, including movies and TV shows, with features such as genre, cast, director, country, release year, rating, duration, and description.

---

## Machine Learning Problem Type

**Unsupervised Learning – Content-Based Recommendation**

The dataset does not include user interaction data such as ratings or watch history. Therefore, there is no target variable, and the problem is formulated as an unsupervised learning task.

---

## Objective

The objective of this lab is to design a content-based recommendation approach that identifies similarities between Netflix titles based on their content features.

The system can recommend movies or TV shows that are similar to a selected title based on learned feature similarity.

---

## Selected Features

- Genre (`listed_in`)  
- Cast  
- Director  
- Country  
- Rating  
- Duration  
- Release Year  
- Description  

Identifiers such as `show_id` and `title` are excluded from similarity modeling.

---

## Notebook Tasks

The notebook includes basic dataset inspection:

- Loading the dataset using Pandas  
- Displaying dataset shape  
- Previewing first rows  
- Listing column names  
- Inspecting data types  

No model training is implemented in this lab.

---

## Methodology Workflow

1. Dataset Selection  
2. Data Loading  
3. Data Inspection  
4. Data Cleaning  
5. Feature Engineering  
6. Text Vectorization (TF-IDF)  
7. Similarity Computation (Cosine Similarity)  
8. Recommendation Output  

The methodology diagram is included in this folder.