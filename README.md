# \# Research project: AI-based Requirement Prioritization Tool

# This repository contains the source code and dataset used in the research work “AI-based Requirement Prioritization for Agile Software Development”.

# 

# \# Overview

# This project proposes an AI-based tool to support software requirement prioritization in Agile Software Development. The system combines Machine Learning and Natural Language Processing techniques to automatically analyze software requirements, estimate their impact, and classify them according to priority. In addition, a Large Language Model (LLaMA 3 via Groq API) is used to provide semantic insights and visual representations of requirements through PlantUML diagrams. The tool aims to support project managers and stakeholders in making informed decisions while reducing the effort required for manual requirement analysis.

# 

# \# Repository Structure

# The repository is organized as follows:

# .

# ├── dataset/

# │   ├── case\_study.csv

# │   └── full\_requirements.csv

# │   └── prioritized\_requirements.csv

# │   └── test.csv

# │   └── test\_labeled.csv

# │   └── train.csv

# │   └── train\_labeled.csv

# ├── models/

# │   ├── all-MiniLM-L6-v2/

# │   └── classifier.pkl

# │   └── encoder.pkl

# │   └── gnn\_model.pt

# │   └── pca.pkl

# │   └── regressor\_cost\_time.pkl

# │   └── scaler.pkl

# ├── scripts/

# │   ├── labeling/

# │   │   └── labeling1.py

# │   │   └── labeling2.py

# │   │   └── labeling3.py

# │   └── classifier.py

# │   └── preprocessing.py

# ├── .env

# ├── ai\_analysis.py

# ├── app.py

# ├── architecture.puml

# ├── cost\_time\_prediction.py

# ├── gnn\_predictor.py

# ├── preprocess\_input.py

# ├── requirements.txt

# └── README.md

# 

# \# Dataset

# The dataset consists of functional and non-functional software requirements describing a generic software system. Requirements cover typical aspects such as usability, availability, performance, and service quality. Each requirement is associated with structured features including urgency, origin of the change, type of change, number of dependencies, estimated cost, estimated implementation time, and priority.

# 

# \# Labeling Process

# With the exception of the functional/non-functional label provided by the original PROMISE dataset, all additional labels were generated automatically using heuristic rules and pre-trained NLP and ML models. These labels do not represent expert-validated ground truth and should be interpreted as indicative labels for research and comparative evaluation purposes only.

# 

# \# Models

# The following methods are implemented and evaluated in this repository:

# \- Random Forest classifier (proposed model)

# \- Rule-based prioritization method (baseline 1)

# \- Decision Tree classifier (baseline 2)

# An ablation study is also included to analyze the contribution of different feature groups to the overall performance of the Random Forest model.

# 

# \# Evaluation

# Models are evaluated using a held-out pre-labeled test set. Since evaluation relies on automatically labeled data, results should be interpreted as relative performance comparisons rather than absolute performance indicators in real-world industrial settings.

# 

# \# How to Run

# 1\. Clone the repository:

# &nbsp;  git clone https://github.com/Mari-eng02/Research\_project\_AI.git

# &nbsp;  cd Research\_project\_AI

# 2\. Install the required dependencies: pip install -r requirements.txt

# 3\. Run the web app: python app.py

# 

# Important: To enable the AI-based semantic analysis using LLaMA 3, a valid Groq API key must be provided as an environment variable.

# 

# \# Reproducibility

# All experiments reported in the associated paper can be reproduced using the code and data provided in this repository. Random seeds are fixed where applicable to improve reproducibility.

# 

# 



