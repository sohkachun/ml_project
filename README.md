## End to End Machine Learning Project 
# Student Performance Prediction Project

## Project Overview
This project focuses on predicting student performance based on various features, using machine learning techniques. The project involves an end-to-end pipeline, starting with Exploratory Data Analysis (EDA), model training, and deployment. Key components of the project include data preprocessing, model building, and the integration of logging and exception handling to ensure robustness.

Through this project, I have gained hands-on experience in the following:
- Exploratory Data Analysis (EDA) and feature engineering.
- Building and training machine learning models (e.g., CatBoost).
- Preprocessing data and saving pre-trained models.
- Structuring projects for machine learning deployment using tools like Docker and Flask.
- Implementing logging and exception handling in Python applications.
- Creating modular code components and a clean project architecture.

## Directory Structure
Here’s an overview of the project structure and the purpose of each file/folder:

### Main Project Files
- **app.py**: The main application file that runs the Flask web server for student performance prediction.
- **requirements.txt**: Lists all the dependencies required to run the project, including Flask, pandas, CatBoost, etc.
- **setup.py**: Script for setting up the project as a Python package.
- **Dockerfile**: Defines the Docker image for deploying the application.
- **README.md**: Project documentation (this file).

### Data and Model Files
- **train.csv**: The training dataset for model building.
- **test.csv**: The test dataset for evaluating the model.
- **data.csv**: A combined dataset used for analysis and modeling.
- **model.pkl**: The saved trained machine learning model (CatBoost).
- **preprocessor.pkl**: The saved data preprocessing pipeline.
- **artifact**: Directory where models and preprocessed data are stored after training.
- **catboost_info**: Directory containing logs and additional information from the CatBoost model.

### Jupyter Notebooks
- **1. EDA STUDENT PERFORMANCE.ipynb**: Notebook containing the exploratory data analysis, where I examined the dataset, visualized features, and performed data cleaning.
- **2. MODEL TRAINING.ipynb**: Notebook detailing the steps taken to train machine learning models, including hyperparameter tuning and evaluation metrics.

### Source Code
- **source**: The source code for the project.
    - **components**: Contains code modules that are part of the machine learning pipeline, such as data preprocessing and model training.
    - **pipeline**: Implements the end-to-end pipeline for data processing and prediction.
    - **__init__.py**: Initializes the source package.
    - **exception.py**: Contains custom exception classes to handle errors gracefully.
    - **logger.py**: Implements logging to track the application’s progress and debug issues.
    - **utils.py**: Utility functions used across the project for tasks such as loading data, saving models, etc.

### Templates
- **templates**: HTML templates for the web application’s user interface.
    - **home.html**: The main page of the web app where users can input data for prediction.
    - **index.html**: The landing page of the web app.

### Additional Configuration Files
- **.ebextensions**: Configuration for deployment on AWS Elastic Beanstalk.
- **python.config**: Python runtime configuration for the application.
- **.gitignore**: Specifies files and directories to be ignored by Git (e.g., `.pkl` files, temporary files).

## Key Learnings

1. **Exploratory Data Analysis (EDA)**
   - Learned how to clean, visualize, and explore data for better feature selection.
   - Gained experience in identifying correlations and patterns in student performance data.

2. **Model Training and Evaluation**
   - Implemented and trained a **CatBoost** model for predicting student performance.
   - Tuned hyperparameters for optimal model performance and saved the trained model as a pickle file.
   - Evaluated model accuracy using metrics like RMSE, MAE, and R².

3. **Data Preprocessing**
   - Developed a preprocessing pipeline that includes data normalization, feature encoding, and missing value imputation.
   - Serialized the preprocessing pipeline using **pickle** to ensure consistency between training and prediction.

4. **Building a Modular Python Application**
   - Structured the project using modular Python files such as `exception.py`, `logger.py`, and `utils.py`, which handle errors, logging, and reusable utilities.
   - Encapsulated the machine learning logic in components and pipelines for easy maintenance and scalability.

5. **Deploying a Flask Web Application**
   - Created a simple web interface using **Flask** that allows users to input student data and get performance predictions.
   - Implemented form handling and integrated the trained model for real-time predictions.

6. **Using Docker for Deployment**
   - Learned how to containerize the application using **Docker** for consistent deployment across environments.
   - Built a Docker image using a `Dockerfile` that defines all dependencies and the environment setup.

7. **Version Control with Git and GitHub**
   - Managed project code, datasets, and artifacts using **Git** for version control.
   - Ignored unnecessary files using `.gitignore` to keep the repository clean.

## Running the Project

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   cd student-performance-prediction
