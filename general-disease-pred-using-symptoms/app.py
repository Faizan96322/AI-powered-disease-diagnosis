import streamlit as st
import numpy as np
import pickle
import pandas as pd

with open('model','rb') as f:
    model,diseases,sym_attrs = pickle.load(f)

def predict_disease(symptoms):
    sym_indices = [np.where(sym_attrs==sym)[0][0] for sym in symptoms]
    x = np.zeros(len(sym_attrs))
    for idx in sym_indices:
        x[idx]=1
    pred = model.predict([x])[0]
    return diseases[pred]

df = pd.read_csv('Prototype.csv')

def get_likely_syms(df,sym_list):
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
    return syms, l2

st.title("Disease Prediction")
lst = [x.replace("_"," ") for x in df.columns]
symptoms = st.multiselect("Please tell us what you feel:",lst)
symptoms = [x.replace(" ","_") for x in symptoms]
if len(symptoms):
    like_syms, s = get_likely_syms(df,symptoms)
    if len(s):
        s = [x.replace("_"," ") for x in s]
        l = st.multiselect("Have you been feeling any of these? If YES please select:",s)
        l = [x.replace(" ","_") for x in l]
        symptoms = symptoms + l
    
    submit = st.button('Diagnose')

    if submit:
        st.success(predict_disease(symptoms))

