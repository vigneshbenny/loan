# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:54:35 2024

@author: LENOVO
"""

import numpy as np
import joblib
d=joblib.load("C:/Users/LENOVO/Downloads/archive (4)/loanpredict.pkl")
input_data=(1,1,3.0,0,0,2,1)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
result=d.predict(input_data_reshaped)
if result==0:
    print(" loan is not approved")
else:
    print("loan is approved")    