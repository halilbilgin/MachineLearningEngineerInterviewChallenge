### Challenge

Explore the data for identified fraudsters and other users. What are your preliminary observations? 

**Databases**

Write an ETL script in Python to load the data into the PostgreSQL database. The associated DDL should be executed through Python and not directly in SQL. You can find the desired schema in schema.yaml and some sample code for the ETL.


**Feature Engineering**

Utilizing your findings from part a) and some creativity, create some features. Explain your reasoning behind the features.
Make a features.py script which when executed will create these features and store them in the DB. 


**Model Selection/Validation**

Create an ML model which identifies fraudsters. Assess the quality of your model and explain.
Make a train.py file which generates the fitted model artifact (it should be stored under the artifacts sub-directory).


**Operationalization**

How will you utilize this model to catch fraudsters? If a fraudster is identified, what should be the resulting action: LOCK_USER, ALERT_AGENT, or BOTH? Explain.
Make a patrol.py file and write a simple function which implements your logic from above. The function should accept a user_id and yield the suggested action(s) (e.g. patrol(user_id) = [‘LOCK_USER’, ‘ALERT_AGENT’])


### My solution

Database : [`code/etl.py`](/code/etl.py) 

Feature engineering: [`code/features.py`](/code/features.py)

Model selection and validation: [`code/train.py`](/code/train.py) 

Operationalization: [`code/patrol.py`](/code/patrol.py)

Here is the jupyter notebook, where I presented all my solution. [Run pipeline.ipynb](https://github.com/halilbilgin/MachineLearningEngineerInterviewChallenge/blob/master/notebooks/Run%20pipeline.ipynb)
