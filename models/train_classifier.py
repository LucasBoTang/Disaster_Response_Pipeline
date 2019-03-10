import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score,make_scorer
import pickle
nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_filepath):
    """
    - load cleaned data from sqllite database

    Arg:
        database_filepath: file path of database
    Returns:
        X: messages
        Y: categories of disaster
        category_names: name of  categories
    """
    # load data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    engine.dispose()

    # extract X and Y
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    - tokenize text

    Arg:
        text: message string
    Return:
        token: list of tokens
    """
    # convert to lowercase
    text = text.lower()
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    token = nltk.tokenize.word_tokenize(text)
    # remove stop words
    token = [word for word in token if word not in nltk.corpus.stopwords.words('english')]
    # lemmatize
    token = [nltk.stem.wordnet.WordNetLemmatizer().lemmatize(word) for word in token]
    return token


def build_model():
    """
    - build model

    Return:
        pipeline: machine learning pipeline
    """
    # set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multioutput', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    - evaluate model

    Args:
        model: machine learning pipeline
        X_test: test set of X
        Y_test: test set of Y
        category_names: name of  categories
    """
    # predict
    Y_pred = model.predict(X_test)

    # evaluate
    for i in range(len(category_names)):
        cat = category_names[i]
        print(cat)
        print(classification_report(Y_test[cat], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    - export model as a pickle file

    Args:
        model: fitted machine learning pipeline
        model_filepath: file path of pickle file
    """
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
