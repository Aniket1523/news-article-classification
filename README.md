
# News Article Classification

In this project, we aim to build a robust news article classification model using machine learning techniques. The process involves several key steps, including data loading, preprocessing, feature extraction, and model training. We utilize a Random Forest Classifier and employ hyperparameter tuning through GridSearchCV to enhance the model's performance. The dataset consists of news articles from various categories, and we implement a text classification approach to categorize articles into predefined topics such as business, entertainment, politics, sports, and technology. The ultimate goal is to achieve high accuracy and precision in classifying articles, providing a valuable tool for efficiently organizing and categorizing large volumes of news content.

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


## Project Structure


- `notebooks/`: Contains Jupyter notebook of the project.
- `bbc/`: Stores the text articles with 5 different categories
- `requirements.txt`: Lists all the dependencies needed for the project.
- `README.md`: The main documentation file.


## Installation


1.Clone the repository to your local machine:
git clone https://github.com/Aniket1523/news-article-classification

2.Navigate to the project directory:
cd news-article-classification

3.In the Jupyter interface, open the notebook file news-article-classification.ipynb.

4.Change the 'main_data_dir' to the path where you have placed downloaded dataset.

5.Kindly make sure you are downloading and importing all the libararies reuired by running very first section of notebook.

## Usage/Examples

To explore the project and run the code, make sure you have [Jupyter Notebook](https://jupyter.org/install) installed on your machine. Open the `your_project_notebook.ipynb` file using Jupyter Notebook, and run the cells sequentially to reproduce the results.

    

## Results

After conducting extensive experiments and fine-tuning our news article classification model, we achieved promising results. The initial baseline model, a Random Forest Classifier, demonstrated a commendable accuracy of 95.8% on the test set. However, the optimization led to further improvements through hyperparameter tuning using GridSearchCV.

The best-performing model, identified through the grid search process, achieved an outstanding accuracy of 97.2% on the test data. The classification report provides a detailed breakdown of precision, recall, and F1-score metrics for each news category, including business, entertainment, politics, sport, and tech. These metrics collectively highlight the robustness and effectiveness of the tuned Random Forest Classifier.

In summary, our model showcases remarkable performance in accurately classifying news articles, providing a solid foundation for text classification tasks.