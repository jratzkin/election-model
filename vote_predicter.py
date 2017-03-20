
import pandas as pd
import pylab as P
from sklearn import linear_model

################
## load data  ##
################

## this is from the 2010 census report

CA_sex_df = pd.read_csv('data/US_census_data/2010_census/CA_sex_by_tract.csv')
CA_race_df = pd.read_csv('data/US_census_data/2010_census/CA_race_by_tract.csv')
CA_age_df = pd.read_csv('data/US_census_data/2010_census/CA_age_by_tract.csv')

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

#####################################
##  form new data frames to model  ##
#####################################


num_CA_counties = CA_pres_results.shape[0]

num_CA_tracts = CA_race_df.shape[0]

CA_df = pd.DataFrame()
CA_df['county'] = CA_race_df['COUNTY']
CA_df['tract'] = CA_race_df['NAME']
CA_df['total population'] = CA_race_df['POP100']
CA_df['median age'] = CA_age_df['P013001']
CA_df['median male age'] = CA_age_df['P013002']
CA_df['median female age'] = CA_age_df['P013003']
CA_df['male population'] = CA_sex_df['P012002'] / CA_sex_df['POP100']  # normalized
CA_df['female population'] = CA_sex_df['P012026'] / CA_sex_df['POP100'] #normalized
CA_df['white pop'] = CA_race_df['P009005']/CA_race_df['POP100'] #normalized
CA_df['latino pop'] = CA_race_df['P009002']/CA_race_df['POP100'] #normalized
CA_df['african-am pop'] = CA_race_df['P009006']/CA_race_df['POP100'] #normalized
CA_df['native-am pop'] = CA_race_df['P009007']/CA_race_df['POP100'] #normalized
CA_df['asian pop'] = CA_race_df['P009008']/CA_race_df['POP100'] #normalized
CA_df['pac islander pop'] = CA_race_df['P009009']/CA_race_df['POP100'] #normalized
CA_df['other single race pop'] = CA_race_df['P009010']/CA_race_df['POP100'] #normalized
CA_df['mixed race pop'] = CA_race_df['P009011']/CA_race_df['POP100'] #normalized

CA_df['trump vote'] = 0
CA_df['clinton vote'] = 0
CA_df['other vote'] = 0
for i in range(num_CA_counties):
    mask = CA_df['county'] == 2*i + 1
    trump_vote = CA_pres_results.iloc[i, 2]
    clinton_vote = CA_pres_results.iloc[i,3]
    other_vote = 100 - trump_vote - clinton_vote
    CA_df['trump vote'][mask] = trump_vote
    CA_df['clinton vote'][mask] = clinton_vote
    CA_df['other vote'][mask] = other_vote


CA_array = CA_df.as_matrix()
CA_data = CA_array[:,3:-3]
CA_target = CA_array[:,-3:]
CA_train_data = CA_data[: 0.9*num_CA_tracts, :]
CA_test_data = CA_data[0.9*num_CA_tracts:, :]
CA_train_target = CA_target[: 0.9*num_CA_tracts, :]
CA_test_target = CA_target[0.9*num_CA_tracts:, :]

print ('-' * 30)
print ('modeling voting in California now')
print ('-' * 30)

lin_regr = linear_model.LinearRegression()
lin_regr.fit(CA_train_data, CA_train_target)
#print ('coefficients of linear regression for the CA voter model: \n%r' % lin_regr.coef_)
print ('linear regression score: %f' % lin_regr.score(CA_test_data, CA_test_target))
lin_prediction_CA = lin_regr.predict(CA_test_data)
lin_residuals_CA = lin_prediction_CA - CA_test_target
lin_residuals_CA_trump = lin_residuals_CA[:,0]
lin_residuals_CA_clinton = lin_residuals_CA[:,1]
print ('using linear regression: the mean error in the trump prediction is %f and the mean error in the clinton prediction is %f' % (lin_residuals_CA_trump.mean(), lin_residuals_CA_clinton.mean()))

P.figure()
n, bins, patches = P.hist([lin_residuals_CA_trump, lin_residuals_CA_clinton], bins=50, histtype='bar', color=['red', 'blue'], label=['Trump', 'Clinton'])
P.legend()
P.title('linear regression')
P.show()

alpha = 0.75
ridge_regr = linear_model.Ridge(alpha=alpha)
ridge_regr.fit(CA_train_data, CA_train_target)
#print ('coefficients of ridge regression for the CA voter model: \n%r' % ridge_regr.coef_)
print ('ridge regression score: %f' % ridge_regr.score(CA_test_data, CA_test_target))
ridge_prediction_CA = ridge_regr.predict(CA_test_data)
ridge_residuals_CA = ridge_prediction_CA - CA_test_target
ridge_residuals_CA_trump = ridge_residuals_CA[:,0]
ridge_residuals_CA_clinton = ridge_residuals_CA[:,1]
print ('using ridge regression: the mean error in the trump prediction is %f and the mean error in the clinton prediction is %f' % (ridge_residuals_CA_trump.mean(), ridge_residuals_CA_clinton.mean()))

P.figure()
n, bins, patches = P.hist([ridge_residuals_CA_trump, ridge_residuals_CA_clinton], bins=50, histtype='bar', color=['red', 'blue'], label=['Trump', 'Clinton'])
P.legend()
P.title('Ridge regression')
P.show()

alpha = 0.5
lasso_regr = linear_model.Lasso(alpha=alpha)
lasso_regr.fit(CA_train_data, CA_train_target)
#print ('coefficients of lasso regression for the CA voter model: \n%r' % lasso_regr.coef_)
print ('lasso regression score: %f' % lasso_regr.score(CA_test_data, CA_test_target))
lasso_prediction_CA = lasso_regr.predict(CA_test_data)
lasso_residuals_CA = lasso_prediction_CA - CA_test_target
lasso_residuals_CA_trump = lasso_residuals_CA[:,0]
lasso_residuals_CA_clinton = lasso_residuals_CA[:,1]
print ('using lasso regression: the mean error in the trump prediction is %f and the mean error in the clinton prediction is %f' % (lasso_residuals_CA_trump.mean(), lasso_residuals_CA_clinton.mean()))

P.figure()
n, bins, patches = P.hist([lasso_residuals_CA_trump, lasso_residuals_CA_clinton], bins=50, histtype='bar', color=['red', 'blue'], label=['Trump', 'Clinton'])
P.legend()
P.title('Lasso regression')
P.show()
