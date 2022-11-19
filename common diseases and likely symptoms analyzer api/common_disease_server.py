from fastapi import FastAPI, Query
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import pandas as pd
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

with open('model','rb') as f:
    model,diseases,sym_attrs = pickle.load(f)
df = pd.read_csv('Prototype.csv')

@app.get('/predict')
def predict(symptoms: list[str] = Query(...)):
    sym_indices = [np.where(sym_attrs==sym)[0][0] for sym in symptoms]
    x = np.zeros(len(sym_attrs))
    for idx in sym_indices:
        x[idx]=1
    pred = model.predict([x])[0]
    return diseases[pred]

@app.get('/get_lksyms')
def get_likely_symptoms(sym_list: list[str] = Query(...)):
    syms = []
    for sym in sym_list:
        sl = {}
        for s in df.columns[:-1]:
            if (s in sym_list):
                continue
            try:
                sl[s] = pd.crosstab(df[sym],df[s])[1][1]
            except:
                print(sym,s)
        sl = sorted(sl.items(),key = lambda x:x[1], reverse=True)
        y = [x[0] for x in sl if x[1]>0] #if x[1]>0.1*df[df[sym]==1].shape[0] 
        syms.append(y)
    #syms = sorted(syms.items(),key = lambda x:x[1], reverse=True)
    #print(syms)
    #count = int(np.mean([df[df[x]==1].shape[0] for x in sym_list]))
    l = syms[0]
    l2=[]
    for s in l:
        flag=1
        for lst in syms[1:]:
            if s in lst:
                flag=1
            else:
                flag=0
                break
        if flag:
            l2.append(s)
    js_l2 = json.dumps(l2)
    return l2


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)