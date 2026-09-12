# Fake News Detection Using Machine Learning

## Project Overview

Fake news has become a significant problem with the rapid growth of online news platforms and social media. Misleading or fabricated news articles can spread quickly and make it difficult for people to distinguish reliable information from false information.

This project develops a **Fake News Detection System** that classifies news articles as **Fake or Genuine** using their headline and body text. The project uses the **Kaggle Fake News Dataset and LIAR Dataset** and applies text preprocessing and **TF-IDF** feature extraction to convert news content into numerical features.

Supervised machine learning algorithms such as **Logistic Regression and Passive-Aggressive Classifier** are used to train and compare classification models. Their performance is evaluated using appropriate classification metrics to determine how effectively they distinguish between fake and genuine news.

The project also discusses an important limitation of text-based fake news detection. Linguistic cues such as word choice, writing style, and sentence structure can provide useful signals, but they cannot always determine whether an article is factually true. Therefore, the system provides a machine learning-based prediction rather than complete factual verification.

## Objectives

* To develop a machine learning system for detecting fake and genuine news.
* To preprocess and clean news headlines and article content.
* To convert textual data into numerical features using TF-IDF.
* To apply supervised machine learning classification techniques.
* To compare the performance of different classification models.
* To evaluate the models using suitable classification metrics.
* To understand the limitations of text-based fake news detection.

## Dataset

The project uses two datasets:

### Kaggle Fake News Dataset

The Kaggle Fake News dataset contains news articles that can be used to train a model to distinguish between fake and genuine news.

### LIAR Dataset

The LIAR dataset contains statements and their associated truthfulness labels. It can be used to evaluate machine learning approaches for detecting misleading or false information.

The datasets are used for data preprocessing, feature engineering, model training, validation, and testing.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Jupyter Notebook
* Google Colab
* VS Code

## Machine Learning Techniques

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert news text into numerical features. It represents the importance of words within a document and across the collection of documents.

### Logistic Regression

Logistic Regression is a supervised learning algorithm used for classification. It provides a baseline model for distinguishing between fake and genuine news articles.

### Passive-Aggressive Classifier

The Passive-Aggressive Classifier is a supervised classification algorithm that is particularly useful for text classification. It is trained using the TF-IDF features and compared with Logistic Regression.

## Methodology

The project follows the following machine learning workflow:

```text
Raw News Data
      |
      v
Data Cleaning and Preprocessing
      |
      v
Train / Validation / Test Split
      |
      v
TF-IDF Feature Engineering
      |
      v
Model Training
      |
      +----------------------------+
      |                            |
      v                            v
Logistic Regression       Passive-Aggressive Classifier
      |                            |
      +------------+---------------+
                   |
                   v
             Model Evaluation
                   |
                   v
          Fake / Genuine Prediction
```

## Data Preprocessing

The news text is processed before applying machine learning algorithms. The preprocessing stage may include:

* Handling missing values
* Removing unnecessary characters
* Converting text to a consistent format
* Removing unwanted symbols
* Tokenization
* Removing stop words where appropriate
* Preparing headline and body text for feature extraction

The exact preprocessing steps depend on the structure and quality of the selected dataset.

## Feature Engineering

After preprocessing, the text is converted into numerical features using **TF-IDF**.

The TF-IDF representation allows traditional machine learning algorithms to process textual information and identify patterns that may help distinguish fake news from genuine news.

## Model Training

The processed dataset is divided into appropriate training and testing portions. The TF-IDF features from the training data are used to train the classification models.

The following models are compared:

1. Logistic Regression
2. Passive-Aggressive Classifier

The models are evaluated on unseen test data to understand their classification performance.

## Model Evaluation

The classification models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC
* PR-AUC

These metrics provide a more complete understanding of model performance instead of relying only on accuracy.

For example:

* **Accuracy** measures the overall percentage of correct predictions.
* **Precision** measures how many predicted fake articles are actually fake.
* **Recall** measures how many actual fake articles are correctly identified.
* **F1-Score** provides a balance between precision and recall.
* **Confusion Matrix** shows the different types of correct and incorrect predictions.

## Results

The final results will be added after training and evaluating the models.

Example format:

| Model                         | Accuracy | Precision | Recall | F1-Score |
| ----------------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression           |      XX% |       XX% |    XX% |      XX% |
| Passive-Aggressive Classifier |      XX% |       XX% |    XX% |      XX% |

The model with the better overall performance can be selected for the final prediction system.

## Limitations

This project primarily relies on textual and linguistic features. Although word choice, writing style, and sentence structure can provide useful information, they cannot always determine whether a news article is factually true.

A well-written fake article may appear similar to a genuine article, while a genuine article may contain language that resembles patterns commonly found in fake news. Therefore, linguistic cues alone are not sufficient for complete fact verification.

The system should therefore be considered a **text-based machine learning prediction system rather than a complete fact-checking system**.

## Project Structure

The project can be organized as follows:

```text
Fake-News-Detection/
|
|-- README.md
|-- requirements.txt
|
|-- data/
|   |-- fake_news_dataset
|   |-- liar_dataset
|
|-- notebooks/
|   |-- Fake_News_Detection.ipynb
|
|-- src/
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- train.py
|   |-- predict.py
|
|-- models/
|   |-- trained_model.pkl
|
```

The actual folder structure may be modified according to the files used in the project.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Fake-News-Detection
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Requirements

The main Python libraries used in this project include:

```text
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

Additional libraries should be added to `requirements.txt` if they are used in the final implementation.

## How to Run

1. Download or prepare the required datasets.
2. Place the datasets in the appropriate data directory.
3. Open the `Fake_News_Detection.ipynb` notebook.
4. Run the preprocessing steps.
5. Generate TF-IDF features.
6. Train the Logistic Regression and Passive-Aggressive Classifier models.
7. Evaluate the models using the selected metrics.
8. Use the trained model to classify new news articles as Fake or Genuine.

## Example Prediction

Example input:

```text
News Article:
The government announced a new policy to improve public education.
```

Example output:

```text
Prediction: Genuine
```

The actual prediction depends on the trained model and the dataset used.

## Course Alignment

This project is aligned with the Machine Learning syllabus covered in Modules M1–M6.

### M1 - PlacementPredict as a System: The ML Lifecycle

The project follows the basic machine learning lifecycle:

```text
Raw Data
→ Cleaned Data
→ Feature Engineering
→ Training
→ Model
→ Prediction
```

### M2 - Supervised Learning

The project uses supervised classification techniques, particularly Logistic Regression, for binary classification.

### M3 - Tree-Based Models

Tree-based models such as Random Forest or Gradient Boosting can be considered for additional comparison if implemented in the project.

### M5 - Model Evaluation, Selection, and Calibration

The project applies model evaluation techniques such as train/test splitting and classification metrics including precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix.

### M6 - ML Engineering

The project can be extended from a notebook-based workflow toward a production-oriented ML system through model packaging, serving, monitoring, and retraining.

## Future Improvements

Possible improvements include:

* Adding more diverse and reliable datasets.
* Improving text preprocessing and feature engineering.
* Comparing additional classical machine learning models.
* Handling class imbalance where required.
* Performing hyperparameter tuning.
* Adding cross-validation for more reliable model comparison.
* Deploying the trained model as a prediction service.
* Adding monitoring and retraining capabilities.
* Incorporating external evidence for stronger fact verification.

## Conclusion

This project demonstrates how supervised machine learning and text feature engineering can be used to classify news articles as fake or genuine. TF-IDF is used to represent textual information, while Logistic Regression and Passive-Aggressive Classifier are used for classification.

The project also demonstrates the importance of proper model evaluation and recognizes that text-based linguistic features alone cannot establish the factual truth of a news article. The system therefore serves as a machine learning-based prediction tool and provides a foundation for developing more advanced fake news detection systems.

## Contributors

* Name: Akshara

* Roll No: 2520030269

* Name: sudeshna

* Roll No: 2520030591
  
## License

This project is developed for academic and educational purposes.
