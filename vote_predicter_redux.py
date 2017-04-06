
import pandas as pd
import pylab as P
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

################
## load data  ##
################

## this is from the 2010 census report

CA_sex_df = pd.read_csv('data/US_census_data/2010_census/CA_sex_by_tract.csv')
CA_race_df = pd.read_csv('data/US_census_data/2010_census/CA_race_by_tract.csv')
CA_age_df = pd.read_csv('data/US_census_data/2010_census/CA_age_by_tract.csv')

## from 2015 ACS

df1 = pd.read_csv('data/US_census_data/2015_ACS/ACS_15_5YR_S0101_with_ann.csv') # age and sex data
df2 = pd.read_csv('data/US_census_data/2015_ACS/ACS_15_5YR_DP03_with_ann.csv')  # employment data
df3 = pd.read_csv('data/US_census_data/2015_ACS/ACS_15_5YR_S1501_with_ann.csv') # educational attainment
df4 = pd.read_csv('data/US_census_data/2015_ACS/ACS_15_5YR_B03002_with_ann.csv') # race


## labels: 

# POP100: total population, in all dataframes
# P009002: latino population, in race datafram
# P009005: white population, in race dataframe
# P009006: african american population, in race dataframe
# P009007: native american population, in race dataframe
# P009008: asian population, in race dataframe
# P009009: pacific islander population, in race dataframe
# P009010: other single race population, in race dataframe
# P009011: mixed race population, in race dataframe
# P012002: male population, in sex dataframe
# P012026: female population, in sex dataframe
# P013001: median age, in age dataframe
# P013002: median male age, in age dataframe
# P013003: median female age, in age dataframe

pres_results_df = pd.read_csv('data/election_results/2016/presidential_results_2016_by_county.csv')

###################
## process data  ##
###################

# separate election results by state

CA_pres_results = pres_results_df[pres_results_df.state=='CA']
# assign labels to each county to match the US census labelling 

CA_pres_results['county label'] = 2*CA_pres_results.index + 1

#delete rows with zero total population
CA_race_df = CA_race_df[CA_race_df['POP100'] != 0]
CA_sex_df = CA_sex_df[CA_sex_df['POP100'] != 0]
CA_age_df = CA_age_df[CA_age_df['POP100'] != 0]

df1['Census Tract Number'] = ' '
df1['County'] = ' '
for i in range(df1.shape[0]):
    temp_list = df1.iloc[i,2].split()
    df1.iloc[i,-2] = temp_list[2]
    if len(temp_list)>6:
        county_name = []
        for j in range(3,len(temp_list) - 2):
            county_name.append(temp_list[j])
        df1.iloc[i,-1] = ' '.join(county_name)
    else:
        df1.iloc[i,-1] = temp_list[3]

df2['Census Tract Number'] = ' '
df2['County'] = ' '
for i in range(df2.shape[0]):
    temp_list = df2.iloc[i,2].split()
    df2.iloc[i,-2] = temp_list[2]
    if len(temp_list)>6:
        county_name = []
        for j in range(3,len(temp_list) - 2):
            county_name.append(temp_list[j])
        df2.iloc[i,-1] = ' '.join(county_name)
    else:
        df2.iloc[i,-1] = temp_list[3]

df3['Census Tract Number'] = ' '
df3['County'] = ' '
for i in range(df3.shape[0]):
    temp_list = df3.iloc[i,2].split()
    df3.iloc[i,-2] = temp_list[2]
    if len(temp_list)>6:
        county_name = []
        for j in range(3,len(temp_list) - 2):
            county_name.append(temp_list[j])
        df3.iloc[i,-1] = ' '.join(county_name)
    else:
        df3.iloc[i,-1] = temp_list[3]
        
df4['Census Tract Number'] = ' '
df4['County'] = ' '
for i in range(df4.shape[0]):
    temp_list = df4.iloc[i,2].split()
    df4.iloc[i,-2] = temp_list[2]
    if len(temp_list)>6:
        county_name = []
        for j in range(3,len(temp_list) - 2):
            county_name.append(temp_list[j])
        df4.iloc[i,-1] = ' '.join(county_name)
    else:
        df4.iloc[i,-1] = temp_list[3]

# remove rows with zero total population

zero_list1 = df1[df1['HC01_EST_VC01'] == 0].index.tolist()
zero_list2 = df2[df2['HC01_VC03'] == 0].index.tolist()
zero_list3 = df3[df3['HC01_EST_VC08'] == 0].index.tolist()
zero_list4 = df4[df4['HD01_VD01'] == 0].index.tolist()
zero_list5 = df2[df2['HC01_VC74'] == 0].index.tolist()

zero_list = zero_list1
for i in zero_list2: 
    if i not in zero_list: 
        zero_list.append(i)
for i in zero_list3: 
    if i not in zero_list:
        zero_list.append(i)
for i in zero_list4:
    zero_list.append(i)
for i in zero_list5:
    zero_list.append(i)
    
df1 = df1.drop(df1.index[zero_list])
df2 = df2.drop(df2.index[zero_list])
df3 = df3.drop(df3.index[zero_list])
df4 = df4.drop(df4.index[zero_list])

#####################################
##  form new data frames to model  ##
#####################################


num_CA_counties = CA_pres_results.shape[0]

num_CA_tracts = CA_race_df.shape[0]

CA_df1 = pd.DataFrame()
CA_df1['county'] = CA_race_df['COUNTY']
CA_df1['tract'] = CA_race_df['NAME']
CA_df1['total population'] = CA_race_df['POP100']
CA_df1['median age'] = CA_age_df['P013001']
CA_df1['median male age'] = CA_age_df['P013002']
CA_df1['median female age'] = CA_age_df['P013003']
CA_df1['male population'] = CA_sex_df['P012002'] / CA_sex_df['POP100']  # normalized
CA_df1['female population'] = CA_sex_df['P012026'] / CA_sex_df['POP100'] #normalized
CA_df1['white pop'] = CA_race_df['P009005']/CA_race_df['POP100'] #normalized
CA_df1['latino pop'] = CA_race_df['P009002']/CA_race_df['POP100'] #normalized
CA_df1['african-am pop'] = CA_race_df['P009006']/CA_race_df['POP100'] #normalized
CA_df1['native-am pop'] = CA_race_df['P009007']/CA_race_df['POP100'] #normalized
CA_df1['asian pop'] = CA_race_df['P009008']/CA_race_df['POP100'] #normalized
CA_df1['pac islander pop'] = CA_race_df['P009009']/CA_race_df['POP100'] #normalized
CA_df1['other single race pop'] = CA_race_df['P009010']/CA_race_df['POP100'] #normalized
CA_df1['mixed race pop'] = CA_race_df['P009011']/CA_race_df['POP100'] #normalized

CA_df1['trump vote'] = 0
CA_df1['clinton vote'] = 0
CA_df1['other vote'] = 0
for i in range(num_CA_counties):
    mask = CA_df1['county'] == 2*i + 1
    trump_vote = CA_pres_results.iloc[i, 2]
    clinton_vote = CA_pres_results.iloc[i,3]
    other_vote = 100 - trump_vote - clinton_vote
    CA_df1['trump vote'][mask] = trump_vote
    CA_df1['clinton vote'][mask] = clinton_vote
    CA_df1['other vote'][mask] = other_vote


CA_array1 = CA_df1.as_matrix()
CA_data1 = CA_array1[:,3:-3]
CA_target1 = CA_array1[:,-3:]
CA_train_data1 = CA_data1[: 0.9*num_CA_tracts, :]
CA_test_data1 = CA_data1[0.9*num_CA_tracts:, :]
CA_train_target1 = CA_target1[: 0.9*num_CA_tracts, :]
CA_test_target1 = CA_target1[0.9*num_CA_tracts:, :]

CA_df2 = pd.DataFrame()
CA_df2['County'] = df1['County']
CA_df2['Census Tract Number'] = df1['Census Tract Number']
CA_df2['tot pop'] = df1['HC01_EST_VC01']
CA_df2['male pop'] = df1['HC02_EST_VC01'] / df1['HC01_EST_VC01']
CA_df2['female pop'] = df1['HC03_EST_VC01'] / df1['HC01_EST_VC01']
CA_df2['employment rate'] = df2['HC01_VC04'] / df2['HC01_VC03']
CA_df2['households less than 25,000 per year'] = (df2['HC01_VC75'] + df2['HC01_VC76'] + df2['HC01_VC77'])/df2['HC01_VC74']
CA_df2['households more than 100,000 per year'] = (df2['HC01_VC82'] + df2['HC01_VC83'] + df2['HC01_VC84'])/df2['HC01_VC74']
CA_df2['grad degree'] = df3['HC01_EST_VC15'] / df3['HC01_EST_VC08']
CA_df2['bachelor degree'] = df3['HC01_EST_VC14'] / df3['HC01_EST_VC08']
CA_df2['some college'] = (df3['HC01_EST_VC13'] + df3['HC01_EST_VC12']) / df3['HC01_EST_VC08']
CA_df2['latino pop'] = df4['HD01_VD02'] / df4['HD01_VD01']
CA_df2['white pop'] = df4['HD01_VD03'] / df4['HD01_VD01']
CA_df2['african am pop'] = df4['HD01_VD04'] / df4['HD01_VD01']
CA_df2['native am pop'] = df4['HD01_VD05'] / df4['HD01_VD01']
CA_df2['asian pop'] = df4['HD01_VD06'] / df4['HD01_VD01']
CA_df2['pac islander pop'] = df4['HD01_VD07'] / df4['HD01_VD01']
CA_df2['other race pop'] = df4['HD01_VD08'] / df4['HD01_VD01']

CA_df2['trump vote'] = 0
CA_df2['clinton vote'] = 0
CA_df2['other vote'] = 0
for i in range(CA_pres_results.shape[0]):
    mask = CA_df2['County'] == CA_pres_results['county'][i]
    trump_vote = CA_pres_results.iloc[i, 2]
    clinton_vote = CA_pres_results.iloc[i,3]
    other_vote = 100 - trump_vote - clinton_vote
    CA_df2['trump vote'][mask] = trump_vote
    CA_df2['clinton vote'][mask] = clinton_vote
    CA_df2['other vote'][mask] = other_vote

CA_array2 = CA_df2.as_matrix()

CA_data2 = CA_array2[:,3:-3]
CA_target2 = CA_array2[:,-3:]
CA_train_data2 = CA_data2[: 0.9*(CA_array2.shape[0]), :]
CA_test_data2 = CA_data2[0.9*(CA_array2.shape[0]) :, :]
CA_train_target2 = CA_target2[: 0.9*(CA_array2.shape[0]), :]
CA_test_target2 = CA_target2[0.9*(CA_array2.shape[0]):, :]

################################
###   Run Regression Models  ###
################################

print ('-' * 30)
print ('modeling voting in California using 2010 data now')
print ('-' * 30)

lin_regr = linear_model.LinearRegression()
lin_regr.fit(CA_train_data1, CA_train_target1)
#print ('coefficients of linear regression for the CA voter model: \n%r' % lin_regr.coef_)
print ('linear regression score(2010 data): %f' % lin_regr.score(CA_test_data1, CA_test_target1))
lin_prediction_CA = lin_regr.predict(CA_test_data1)
lin_residuals_CA = lin_prediction_CA - CA_test_target1
lin_residuals_CA_trump = lin_residuals_CA[:,0]
lin_residuals_CA_clinton = lin_residuals_CA[:,1]
print ('using linear regression: the mean error in the trump prediction is %f and the mean error in the clinton prediction is %f (2010 data)' % (lin_residuals_CA_trump.mean(), lin_residuals_CA_clinton.mean()))

lin_scores1 = cross_val_score(lin_regr, CA_data1, CA_target1, cv=5)
print('Five fold cross-validation of the linear regression model (2010 data): \n%r' % lin_scores1)

P.figure()
n, bins, patches = P.hist([lin_residuals_CA_trump, lin_residuals_CA_clinton], bins=50, histtype='bar', color=['red', 'blue'], label=['Trump', 'Clinton'])
P.legend()
P.title('linear regression (2010 data)')
P.show()

alpha = 0.75
ridge_regr = linear_model.Ridge(alpha=alpha)
ridge_regr.fit(CA_train_data1, CA_train_target1)
#print ('coefficients of ridge regression for the CA voter model: \n%r' % ridge_regr.coef_)
print ('ridge regression score (2010 data): %f' % ridge_regr.score(CA_test_data1, CA_test_target1))
ridge_prediction_CA = ridge_regr.predict(CA_test_data1)
ridge_residuals_CA = ridge_prediction_CA - CA_test_target1
ridge_residuals_CA_trump = ridge_residuals_CA[:,0]
ridge_residuals_CA_clinton = ridge_residuals_CA[:,1]
print ('using ridge regression: the mean error in the trump prediction is %f and the mean error in the clinton prediction is %f (2010 data)' % (ridge_residuals_CA_trump.mean(), ridge_residuals_CA_clinton.mean()))

ridge_scores1 = cross_val_score(ridge_regr, CA_data1, CA_target1, cv=5)
print('Five fold cross-validation of the ridge regression model (2010 data): \n%r' % ridge_scores1)


P.figure()
n, bins, patches = P.hist([ridge_residuals_CA_trump, ridge_residuals_CA_clinton], bins=50, histtype='bar', color=['red', 'blue'], label=['Trump', 'Clinton'])
P.legend()
P.title('Ridge regression (2010 data)')
P.show()

print ('-' * 30)
print ('modeling voting in California using 2015 data now')
print ('-' * 30)

lin_regr = linear_model.LinearRegression()
lin_regr.fit(CA_train_data2, CA_train_target2)
#print ('coefficients of linear regression for the CA voter model: \n%r' % lin_regr.coef_)
print ('linear regression score (2015 data): %f' % lin_regr.score(CA_test_data2, CA_test_target2))
lin_prediction_CA = lin_regr.predict(CA_test_data2)
lin_residuals_CA = lin_prediction_CA - CA_test_target2
lin_residuals_CA_trump = lin_residuals_CA[:,0]
lin_residuals_CA_clinton = lin_residuals_CA[:,1]
print ('using linear regression: the mean error in the trump prediction is %f and the mean error in the clinton prediction is %f (2015 data)' % (lin_residuals_CA_trump.mean(), lin_residuals_CA_clinton.mean()))

lin_scores2 = cross_val_score(lin_regr, CA_data2, CA_target2, cv=5)
print('Five fold cross-validation of the linear regression model (2015 data): \n%r' % lin_scores2)

P.figure()
n, bins, patches = P.hist([lin_residuals_CA_trump, lin_residuals_CA_clinton], bins=50, histtype='bar', color=['red', 'blue'], label=['Trump', 'Clinton'])
P.legend()
P.title('linear regression (2015 data)')
P.show()

alpha = 0.75
ridge_regr = linear_model.Ridge(alpha=alpha)
ridge_regr.fit(CA_train_data2, CA_train_target2)
#print ('coefficients of ridge regression for the CA voter model: \n%r' % ridge_regr.coef_)
print ('ridge regression score(2015 data): %f' % ridge_regr.score(CA_test_data2, CA_test_target2))
ridge_prediction_CA = ridge_regr.predict(CA_test_data2)
ridge_residuals_CA = ridge_prediction_CA - CA_test_target2
ridge_residuals_CA_trump = ridge_residuals_CA[:,0]
ridge_residuals_CA_clinton = ridge_residuals_CA[:,1]
print ('using ridge regression: the mean error in the trump prediction is %f and the mean error in the clinton prediction is %f (2015 data)' % (ridge_residuals_CA_trump.mean(), ridge_residuals_CA_clinton.mean()))

P.figure()
n, bins, patches = P.hist([ridge_residuals_CA_trump, ridge_residuals_CA_clinton], bins=50, histtype='bar', color=['red', 'blue'], label=['Trump', 'Clinton'])
P.legend()
P.title('Ridge regression (2015 data)')
P.show()

ridge_scores1 = cross_val_score(ridge_regr, CA_data1, CA_target1, cv=5)
print('Five fold cross-validation of the ridge regression model (2010 data): \n%r' % ridge_scores1)
