# twitter-airline-sentiment

## 1. Project Overview
This project classifies the sentiment of airline-related tweets as positive, neutral, or negative. It covers the complete machine learning workflow, including data preprocessing, model training, evaluation, and deployment as a REST API using FastAPI, Docker, and AWS.

## 2. Project Pipeline 
Raw Tweets -> Text Preprocessing -> Feature Extraction -> Baseline Model Comparison -> PyTorch Baseline -> Fine-Tune BERT -> FastAPI -> Docker -> AWS EC2 Deployment

## 3. Dataset
This project uses the **Twitter US Airline Sentiment** dataset from Kaggle. The dataset contains customer tweets directed at major U.S. airlines, with each tweet manually labeled according to its overall sentiment.

- **Source:** Kaggle – Twitter US Airline Sentiment
- **Number of tweets:** 14,640
- **Number of classes:** 3
  - Negative
  - Neutral
  - Positive

The dataset also includes additional metadata such as the airline name, confidence score, and the reason for negative sentiment. In this project, only the tweet text and sentiment labels are used for model training.

## 4. Exploratory Data Analysis
Before model development, exploratory data analysis (EDA) was performed to better understand the dataset.

The analysis included:

- Distribution of sentiment labels
- Distribution of tweets across airlines
- Most frequent words
- Word clouds for positive, neutral, and negative tweets
- Tweet length analysis

The EDA revealed that the dataset is imbalanced, with negative tweets representing the majority class. To ensure that performance on all three sentiment classes was considered equally, models were evaluated using macro precision, macro recall, and macro F1 score, in addition to accuracy. 

Representative visualizations from the exploratory data analysis are shown below.

### Sentiment Distribution
<img width="582" height="497" alt="Screenshot 2026-07-14 at 12 10 17 AM" src="https://github.com/user-attachments/assets/9915b9ba-1c22-4c8f-9543-8d97b86b61ac" />

### Word Clouds
<img width="507" height="867" alt="Screenshot 2026-07-14 at 12 12 50 AM" src="https://github.com/user-attachments/assets/db1d998b-f0cf-47a4-95f3-378da7d6de3b" />

## 5. Text Preprocessing
The tweet text was cleaned before feature extraction and model training to reduce noise and create a consistent representation of the data. The preprocessing pipeline included the following steps:

- Removed tweets with missing text.
- Converted all text to lowercase.
- Removed URLs and user mentions (`@username`).
- Removed the `#` symbol while preserving the associated hashtag word.
- Removed punctuation and numerical characters.
- Removed extra whitespace and trimmed leading/trailing spaces.
- Applied WordNet lemmatization to convert words to their base verb forms (e.g., *cancelled* → *cancel*).

The original tweet text was preserved separately for error analysis, while the processed text was used for feature extraction and model training.

## 6. Feature Extraction
Two text vectorization techniques were evaluated using **Logistic Regression** with **5-fold cross-validation** to determine the most effective feature representation for the classical machine learning models.

| Feature Extraction | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------------------|---------:|----------------:|-------------:|---------:|
| CountVectorizer | 0.7875 | 0.7385 | **0.7190** | **0.7271** |
| TF-IDF | **0.7915** | **0.7741** | 0.6797 | 0.7133 |

Although TF-IDF achieved slightly higher accuracy and macro precision, **CountVectorizer** produced the highest **Macro F1 score**, which was selected as the primary evaluation metric for comparing feature representations. Therefore, CountVectorizer was used for the subsequent classical machine learning experiments.

## 7. Baseline Classical Machine Learning Models

### Model Selection

After selecting **CountVectorizer** as the feature extraction method, five classical machine learning models were evaluated using **5-fold cross-validation**:

- Logistic Regression
- Naive Bayes
- Linear Support Vector Machine (Linear SVM)
- Random Forest
- XGBoost

Model performance was evaluated using **Accuracy**, **Macro Precision**, **Macro Recall**, and **Macro F1 Score**.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|---------:|----------------:|-------------:|---------:|
| Logistic Regression | **0.7929** | 0.7434 | **0.7274** | **0.7349** |
| Naive Bayes | 0.7687 | 0.7075 | 0.7162 | 0.7112 |
| Linear SVM | 0.7623 | 0.7010 | 0.7005 | 0.7005 |
| Random Forest | 0.7687 | 0.7485 | 0.6360 | 0.6709 |
| XGBoost | 0.7920 | **0.7571** | 0.7015 | 0.7244 |

**Logistic Regression** achieved the highest **Macro F1 score** while also obtaining the highest accuracy and macro recall among the evaluated models. Consequently, Logistic Regression was selected as the strongest classical machine learning baseline for comparison with the subsequent deep learning models.

### Error Analysis

To better understand the strengths and limitations of the baseline model, a confusion matrix, classification report, and misclassified tweets were analyzed.

| Actual / Predicted | Negative | Neutral | Positive |
|--------------------|---------:|--------:|---------:|
| **Negative** | 1522 | 222 | 91 |
| **Neutral** | 209 | 355 | 56 |
| **Positive** | 115 | 84 | 274 |

The confusion matrix shows that **negative tweets were classified most accurately**, achieving an F1 score of **0.83**. Performance was lower for **neutral** (F1 = **0.55**) and **positive** (F1 = **0.61**) tweets, indicating that these classes are more difficult to distinguish.

Inspection of misclassified tweets revealed several common sources of error:

- **Mixed sentiment:** Tweets containing both positive and negative opinions (e.g., flight delays accompanied by praise for the crew).
- **Short or ambiguous messages:** Very short tweets often lacked enough context for reliable classification.
- **Context-dependent language:** Questions, travel updates, and customer service conversations were frequently predicted as neutral despite expressing sentiment.
- **Implicit sentiment:** Some tweets expressed frustration or satisfaction without using strong sentiment words, making them difficult for a bag-of-words model to classify correctly.

These observations highlight a key limitation of classical bag-of-words models: they represent words independently and cannot capture contextual relationships or nuanced language. This motivated the use of a deep learning model, followed by fine-tuning a pretrained BERT model to better model contextual information.

## 8. PyTorch Baseline Model

To establish a deep learning baseline, a simple neural network was implemented in **PyTorch**. The model consists of an **embedding layer**, **mean pooling**, and a **fully connected classification layer**. To improve generalization, **early stopping** was introduced using the **validation Macro F1 score** as the monitoring metric.

- **Patience:** 8 epochs
- **Best validation metric:** Macro F1
- **Training stopped at:** Epoch 28

The best-performing model weights were restored before evaluation on the test set.

### Final Performance

| Metric | Score |
|--------|------:|
| Accuracy | **0.76** |
| Macro Precision | **0.69** |
| Macro Recall | **0.70** |
| Macro F1 | **0.69** |

The PyTorch baseline did not outperform the best classical machine learning model (Logistic Regression + CountVectorizer). One possible reason is that the baseline neural network used a simple architecture consisting of an embedding layer, mean pooling, and a linear classifier trained from scratch. In contrast, CountVectorizer combined with Logistic Regression provides a strong baseline for text classification by effectively leveraging informative word frequencies in a dataset of this size. To better capture the context and meaning of each tweet, the next step was to fine-tune a **pretrained BERT model**.

## 9. Fine-Tuned BERT

To improve upon the classical machine learning and PyTorch baseline models, a **BERT (`bert-base-uncased`)** model was fine-tuned using the **Hugging Face Transformers** library. Unlike the previous models, BERT leverages pretrained contextual language representations, allowing it to better understand the meaning of words within the context of each tweet.

The dataset was split into **training**, **validation**, and **test** sets. Tweets were tokenized using the pretrained BERT tokenizer, and the model was fine-tuned for sentiment classification. The best model was selected based on the **validation Macro F1 score**.

### Test Set Performance

| Metric | Score |
|--------|------:|
| Accuracy | **0.82** |
| Macro Precision | **0.76** |
| Macro Recall | **0.79** |
| Macro F1 | **0.77** |

### Class-wise Performance

| Class | Precision | Recall | F1 Score |
|-------|----------:|-------:|---------:|
| Negative | 0.91 | 0.87 | 0.89 |
| Neutral | 0.67 | 0.63 | 0.65 |
| Positive | 0.69 | 0.87 | 0.77 |

### Discussion

The fine-tuned BERT model achieved the best overall performance among all models evaluated in this project, improving the **Macro F1 score from 0.71 (PyTorch baseline) to 0.77** and the **accuracy from 0.77 to 0.82**. The largest improvements were observed for the **positive** sentiment class, where BERT achieved a recall of **0.87**, demonstrating its ability to better capture contextual information than the classical machine learning models and the simple neural network baseline.

These results demonstrate the advantage of transfer learning for NLP tasks. By starting from a pretrained language model and fine-tuning it on the airline sentiment dataset, BERT was able to learn more robust representations of tweet semantics and produce the strongest sentiment classifier in this project.


## 10. FastAPI

The fine-tuned BERT model was deployed as a REST API using **FastAPI**. The API loads the saved model and tokenizer at startup and exposes a prediction endpoint that accepts raw tweet text and returns the predicted sentiment.



### API Endpoint

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/predict` | Predicts the sentiment of an input tweet. |



### Interactive API Documentation

FastAPI automatically generates interactive API documentation using **Swagger UI**, allowing users to test the API directly from a web browser.

<img width="877" height="802" alt="swagger_docs" src="https://github.com/user-attachments/assets/6d993e5d-bb30-4f2a-b5e5-3447b4171d19" />



### Prediction Example

<img width="823" height="534" alt="prediction_example" src="https://github.com/user-attachments/assets/062e7952-e913-45b7-a51e-259a3acd891b" />




## 11. Docker
To simplify deployment and ensure a reproducible environment, the FastAPI application was containerized using **Docker**.
The Docker image packages:

- The FastAPI application
- The fine-tuned BERT model and tokenizer
- All required Python dependencies
- The application startup configuration

The application can be built and run using:

```bash
docker build -t bert-sentiment-api .
docker run -p 8000:8000 bert-sentiment-api
```

Containerization ensures that the application can be deployed consistently across different environments without requiring manual dependency installation.




## 12. AWS Deployment

The Dockerized application was deployed to **Amazon EC2**.

Deployment workflow:

```
Fine-Tuned BERT
        ↓
Docker Image
        ↓
Amazon ECR
        ↓
Amazon EC2
        ↓
FastAPI REST API
```

The deployment process consisted of:

1. Building the Docker image locally.
2. Pushing the image to Amazon Elastic Container Registry (ECR).
3. Launching an EC2 instance.
4. Pulling the Docker image from ECR.
5. Running the FastAPI application inside a Docker container.

The deployed API successfully served real-time sentiment predictions through the REST endpoint.

### EC2 Instance

<img width="1327" height="67" alt="ec2_instance" src="https://github.com/user-attachments/assets/c54ddab1-ce84-4869-9260-08390992b082" />



### ECR Repository

<img width="1159" height="104" alt="ecr_repository" src="https://github.com/user-attachments/assets/8208c15d-2130-4bf0-bf75-0096c354b491" />



## 13. Results

The project demonstrates a complete machine learning workflow, from data exploration to production deployment.

### Summary

- Performed exploratory data analysis on the Airline Twitter Sentiment dataset.
- Applied text preprocessing and feature engineering.
- Compared CountVectorizer and TF-IDF using Logistic Regression.
- Evaluated five classical machine learning models using 5-fold cross-validation.
- Built a PyTorch baseline neural network and improved its performance using early stopping.
- Fine-tuned a pretrained BERT model using Hugging Face Transformers.
- Developed a REST API with FastAPI.
- Containerized the application with Docker.
- Deployed the API to AWS EC2.

### Best Model Performance

| Metric | Score |
|--------|------:|
| Accuracy | **0.82** |
| Macro Precision | **0.76** |
| Macro Recall | **0.79** |
| Macro F1 | **0.77** |

The fine-tuned BERT model achieved the strongest performance among all models evaluated in this project.


## 14. Future Improvements

Future work could focus on further improving model performance through more extensive hyperparameter optimization using GridSearchCV or Optuna. It would also be interesting to compare the fine-tuned BERT model with other pretrained transformer models, such as RoBERTa and DistilBERT, to evaluate the trade-offs between prediction accuracy and computational efficiency.





