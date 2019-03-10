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

The data in this project comes from [Figure Eight](https://www.figure-eight.com/) - [Multilingual Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/).

This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

Data includes 2 csv files:
* disaster_messages.csv: Messages data.
* disaster_categories.csv: Disaster categories of messages.

## Folder Structure

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- notebook
|- 01ETL_Pipeline_Preparation.ipynb
|- 02ML_Pipeline_Preparation.ipynb

- screenshot
|- web_screenshot1.png
|- web_screenshot2.png

- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
