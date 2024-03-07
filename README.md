# News Article Classification

In this project, we aim to build news article classification model using machine learning techniques. The process involves several key steps, including data loading, preprocessing, feature extraction, and model training. We train a Random Forest Classifier and use hyperparameter tuning through GridSearchCV to enhance the model's performance. The dataset consists of news articles from various categories, and we implement a text classification approach to categorize articles into predefined topics such as business, entertainment, politics, sports, and technology. The ultimate goal is to achieve high accuracy and precision in classifying articles in all the categories. 


## Getting Started
These instructions will guide you through setting up the project on your local machine for development and testing purposes. Ensure you have the required dependencies installed. You can install them by running:

pip install -r requirements.txt

### Prerequisites
scikit-learn
nltk
textblob
spacy
pandas
matplotlib
matplotlib
en_core_web_sm
To explore the project and run the code, make sure you have [Jupyter Notebook](https://jupyter.org/install) installed on your machine.

## Project Structure
- `notebooks/`: Contains Jupyter notebook of the project.
- `bbc/`: Stores the text articles with 5 different categories
- `requirements.txt`: Lists all the dependencies needed for the project.
- `README.md`: The main documentation file.


## Set up and Usage
1.Clone the repository to your local machine:
git clone https://github.com/Aniket1523/news-article-classification or use zip file provided.

2.Navigate to the project directory:
cd news-article-classification

3.Open terminal using 'jupyter notebook' command, which will launce notebook interface.

4.Open the notebook file news-article-classification.ipynb.

4.Change the 'main_data_dir' to the path where you have placed downloaded dataset folder bbc.

5.Kindly make sure you have downloaded and imported all the libararies reuired by running very first section of notebook.

6.Please run all the code sections sequentially as some of the results are dependent on output from previous code sections.
    
## Methodology

### Step 1: Importing Libraries and Loading Dataset

- Import essential libraries and load the dataset.
- Navigate through category folders within `main_data_dir`.
- Read the content of each article and organize the data into a Pandas data frame, associating each text article with its respective category.

### Step 2: Exploratory Data Analysis (EDA)

- Check for duplicate articles and remove them using the 'check_remove_dupli' function.
- Analyze the distribution of unique articles per category through a histogram to assess class balance.

### Step 3: Data Preparation and Splitting

- Split the dataset into training and test sets.
- Reserve the training set for model training and utilize the test set for evaluating the model's performance on unseen data.
- No separate validation set is created initially as it will be used later for hyperparameter tuning.

### Step 4: Feature Extraction

- Use TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.
- Preprocess the text data, including punctuation removal, stopword elimination, tokenization, lemmatization, and lowercase conversion.
- Transform the preprocessed text into TF-IDF matrices for both training and test datasets using the 'TfidfVectorizer' from scikit-learn.

### Step 5: Feature Selection

- Apply 'SelectKBest' with the chi-squared scoring function for feature selection.
- Identify and retain the most informative words that significantly contribute to the classification task.
- Store the selected features in Pandas DataFrames, namely 'X_train_selected_df' and 'X_test_selected_df'.

### Step 6: Sentiment Analysis Feature Creation

- Utilize the TextBlob library to evaluate the sentiment polarity of each text in both training and test datasets.
- Apply the 'create_sentiment_arrays' function to store sentiment scores in separate arrays and convert them into Pandas DataFrames.

### Step 7: Named Entity-based Features

- Use spaCy NLP library to extract named entities from the text data.
- Calculate the frequency of specified entity types within each document.
- Normalize entity frequencies by the total number of entities in the document.
- Store resulting entity frequencies in Pandas DataFrames, creating 'train_entity_df' and 'test_entity_df'.

### Step 8: Combined Feature Sets

- Combine selected TF-IDF features, sentiment scores, and entity frequencies to form combined feature sets for both training and testing.
- Resulting 'X_train_final' and 'X_test_final' are ready for training and evaluating the machine learning model.

### Step 9: Model Training and Evaluation

- Use Random Forest Classifier for model training, creating an ensemble of decision trees.
- Evaluate the baseline model's accuracy on the test set.

### Step 10: Hyperparameter Tuning with GridSearchCV

- Utilize GridSearchCV for hyperparameter tuning with 3-fold cross-validation.
- Explore 243 candidates (combinations of hyperparameters) for optimal model performance.
- Identify the model with the best parameters.

### Step 11: Model Optimization

- Implement the optimized Random Forest Classifier with the identified hyperparameters.
- Achieve a higher accuracy of 97.2% on the test set.


## Results

After conducting extensive experiments and fine-tuning our news article classification model, we achieved promising results. The initial baseline model, a Random Forest Classifier, demonstrated a commendable accuracy of 95.8% on the test set. However, the optimization led to further improvements through hyperparameter tuning using GridSearchCV.

The best-performing model, identified through the grid search process, achieved an outstanding accuracy of 97.2% on the test data. The classification report provides a detailed breakdown of precision, recall, and F1-score metrics for each news category, including business, entertainment, politics, sport, and tech. These metrics collectively highlight the robustness and effectiveness of the tuned Random Forest Classifier.

In summary, our model showcases remarkable performance in accurately classifying news articles, providing a solid foundation for text classification tasks.
