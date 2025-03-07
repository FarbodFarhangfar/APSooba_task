# APSooba_task

the project is in 3 sections

## EDA and trianing

you can find full explaination and code in interview_APSooba.ipynb

and each cell runs by itesel

two models were tested:
XGBoost with Quantile Regression
ayesian Neural Network (BNN)

because there were no enough data for deep learning model and would result in overfitting
I choose the XGBoost with Quantile Regression

## fastapi app and inference

there is a docker based fastapi app that can predect based on input

`docker compose up `

or if you dont want docker , first install requirment.txt

` pip install -r requirements.txt`

go to app directory , run fastapi app with

`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

but I would recommend run with docker

you can predict in url /predict_xgboost

## streamlit interactive dashboard

first install streamlit

`pip install streamlit`

there is an interactive dashboard and you can access it with

`streamlit run .\dashboard.py`
