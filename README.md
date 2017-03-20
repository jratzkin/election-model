# election-model
This code predicts the 2016 presidential election results by county in California

I used data age, sex, and race deomgraphic data from the 2010 census to train three regression models: linear regression (aka least squares fit), Ridge regression, and Lasso regression. It seems the linear regression modeled the data the best, though I would personally rate the performance of all the models as poor. Additionally, all models over-estimated Clinton's preformance and under-estimated Trump's performance. 

In the future I'd like to improve this model. A first step would be to use data from the 2015 ACS, and incorporate some other variables (such as income and education). I expect these new variables would improve performance. Once I have a working model for the California vote I would like to expand to other states. 
