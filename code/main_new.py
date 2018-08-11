import pandas as pd
import xgboost as xgb
import numpy as np
import xlrd
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
time=datetime.now().strftime('%Y%m%d_%H%M%S')
path=os.path.abspath(os.path.dirname(__file__))


data=os.path.join(path,'../Income Statement/Income Statement.xls')
filename='../submit/submit_'+time+'.csv'
submit=os.path.join(path,filename)


wb = xlrd.open_workbook(data)
sheets = wb.sheet_names()
result=pd.DataFrame(columns=['ID', 'predict', 'TICKER_SYMBOL'])

for i in sheets:
    model = xgb.XGBRegressor(max_depth=4, learning_rate=0.1)
    Df = pd.read_excel(data,sheetname=i, converters={u'TICKER_SYMBOL': str})
    df = Df[Df['FISCAL_PERIOD'] == 6]
    df = df[[c for c in df if df[c].isnull().sum() < 10]]
    df = df.sort_values(['PARTY_ID', 'END_DATE'])
    df = df.reset_index(drop=True)
    df['ID'] = df['TICKER_SYMBOL'] + '.' + df['EXCHANGE_CD']
    df['next_revenue'] = df['REVENUE'].shift(-1)
    for i in range(len(df) - 1):
        if df['PARTY_ID'][i] != df['PARTY_ID'][i + 1]:
            df['next_revenue'][i] = np.nan
    print(df[['PARTY_ID', 'next_revenue']])
    data = df[~df['next_revenue'].isnull()]
    goal = df[df['next_revenue'].isnull()]
    train, test = train_test_split(data, test_size=0.2)
    cols = [col for col in df.columns if
            col not in ['EXCHANGE_CD', 'next_revenue', 'PUBLISH_DATE', 'END_DATE_REP', 'END_DATE', 'REPORT_TYPE', 'ID']]
    model.fit(train[cols].astype(float), train['next_revenue'].astype(float))
    predict = model.predict(test[cols].astype(float))

    check = test
    check['predict'] = predict
    print(check)

    model.fit(data[cols].astype(float), data['next_revenue'].astype(float))
    goal_predict = model.predict(goal[cols].astype(float))

    goal['predict'] = goal_predict
    goal = goal[['ID', 'predict', 'TICKER_SYMBOL']]
    print(goal)
    result=result.append(goal)

result.sort_values(by='TICKER_SYMBOL',inplace=True)

result=result[['ID','predict']]
result['predict']=round(result['predict']/1000000,2)

result.to_csv(submit,index=False,header=False)
print('result')
print(result)