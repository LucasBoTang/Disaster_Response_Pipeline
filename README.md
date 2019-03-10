# Data Scientist Nanodegree
# Data Engineering
## Project 5: Disaster Response Pipeline

### Installation:

This project requires **Python 3.x** and the following Python libraries installed:

- [Pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [NLTK](https://www.nltk.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Plotly](https://plot.ly/)
- [Flask](http://flask.pocoo.org/)

### Data:

* disaster_messages.csv: Messages data.
* disaster_categories.csv: Disaster categories of messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
