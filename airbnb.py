import pandas as pd  
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings  
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score

pd.set_option('display.float_format', lambda x : '%.3f' % x)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
plt.rc('axes.spines', right = False, top = False)


### EDA

abnb = pd.read_csv('C:/PYTHON/regr_project/Aemf1.csv')

abnb[abnb.duplicated()] ### There are no duplicate values in the dataframe.

x = 0
for column, no_of_missing_values in abnb.isna().sum().items():
    if no_of_missing_values > 0:
        print('A column with the name of', column, 'is missing', no_of_missing_values, 'values.')
        x += 1 
if x == 0:
    print('All columns have complete data with no missing values, meeting the no-NA condition for Linear Regression.')  ### The dataframe contains no NaN values, satisfying a crucial condition for Linear Regression.
    
#abnb.info()
abnb[['Shared Room', 'Private Room', 'Superhost']] = abnb[['Shared Room', 'Private Room', 'Superhost']].astype(int)    ### Convert boolean values in the dataframe to integer type.
abnb[['City', 'Day', 'Room Type']] = abnb[['City', 'Day', 'Room Type']].astype('category')  ### Convert object-type features with repeated values to category data type to optimize memory usage.

#sns.histplot(abnb['Price']); 
#sns.scatterplot(data = abnb, x = 'City Center (km)', y = 'Price', alpha = 0.65, edgecolor='black', size = 10, linewidth=0.5);
#sns.boxplot(abnb['Price']); ### Charts reveal outliers in the dependent variable 'Price,' with a few daily rental prices exceeding 5000 (EUR?). These may indicate dataset discrepancies or represent flat types the author was regrettably forced to filter out when using Airbnb or Booking during his travels.

abnb['Price_ix'] = abnb['Price'] / abnb.groupby('City')['Price'].transform('mean') 
abnb['City_avg_price'] = abnb.groupby('City')['Price'].transform('mean')  ### Calculation of two metrics: 'Price_ix' and 'City_avg_price'. 'Price_ix' is calculated by dividing the flat's rental price by the average Airbnb rental price in its city. 'City_avg_price' represents the average price per city, used for exploratory data analysis (EDA) purposes.

labels = ['0-1.5', '1.5-3', '3-4.5', '4.5-6', '6-7.5', '7.5-9', '9-10.5', '10.5+' ]
bins = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, abnb['Price_ix'].max()]
abnb['Price_cat'] = pd.cut(abnb['Price_ix'], bins = bins, labels = labels) ### A new column, 'Price_cat', is created using the pd.cut() method to categorize rows based on the flat rental price relative to the average city price. This enables analysis of the distribution of offers with prices significantly higher than the city average.

features_to_an = ['Price', 'City_avg_price', 'Price_ix', 'Price_cat', 'City', 'Room Type', 'Person Capacity', 'Business', 'Guest Satisfaction', 'City Center (km)', 'Attraction Index']
abnb[features_to_an].sort_values('Price_ix', ascending = False).head(25)
abnb.groupby('Price_cat')['Price'].count()#.plot(kind='bar');  ### The majority of rows (88.3%) have a rental price to the average rental price in the city ratio between 0 and 1.5. 10.6% of observations fall within the range of 1.5 to 3. 124 observations (0.3%) have a rental price that is more than 4.5 times the average rental price in that city, and these will be excluded from further analysis. There are 334 observations with a rental price-to-average price ratio between 3 and 4.5, which will be retained for the analysis (at least for now).

abnb = abnb[abnb['Price_ix'] <= 4.5].drop({'Price_ix', 'City_avg_price', 'Price_cat'}, axis = 1)   ### The database has been filtered: observations with extremely high flat rental price relative to the average city price have been removed, and columns redundant for further EDA have been dropped.

abnb['Price'] = abnb['Price'].round(2)

abnb['City'].value_counts(dropna = False).sort_values(ascending = False)#.plot(kind = 'bar'); ### The dataframe contains data from 9 European cities. Amsterdam has the fewest observations (2072), while Rome has the most (9001), making a feature suitable for analysis and inclusion in the model via the pd.get_dummies() function. The average prices vary significantly between cities, with Amsterdam having the highest average price (558.5) and Athens the lowest (147.0), representing a nearly 280% difference. Interestingly, average prices in Lisbon, Vienna, and Berlin are nearly identical (abnb.groupby('City')['Price'].mean().sort_values(ascending = False).plot(kind = 'bar');).

abnb['Weekend'] = np.where(abnb['Day'] == 'Weekend', 1, 0)
abnb.drop('Day', axis = 1, inplace = True)
abnb['Weekend'].value_counts(dropna = False).sort_values()#.plot(kind = 'bar');  ### Firstly, a new column, Weekend, was created to indicate weekends (1) and weekdays (0), replacing the original Day column, which only had two possible values: Weekend and Weekday. The values in the new column are nearly balanced (0/1). However, this column is unlikely to be a strong price predictor, as the average prices for weekends (256.5) and weekdays (248.5) are nearly identical. Notably, larger Weekend vs Weekday price differences are observed in a few cities (e.g., Amsterdam, Budapest), while in Athens and Paris, Weekday prices are slightly higher on average (abnb.groupby('Weekend')['Price'].mean().plot(kind = 'bar'); || abnb.groupby(['City','Weekend'])['Price'].mean().plot(kind = 'bar');

abnb['Room Type'].value_counts(dropna = False).sort_values(ascending = False)#.plot(kind = 'bar'); ### There are only 316 observations for 'Shared Room' compared to over 13k and 28k for other room types. The average price differs significantly across room types. The limited data for 'Shared Room' warrants consideration for removal from further analysis. Additionally, 'Room Type' is likely a strong price predictor (abnb.groupby('Room Type')['Price'].mean().plot(kind = 'bar');).
abnb.groupby(['Room Type', 'Private Room', 'Shared Room'])['Price'].count()  ### The columns 'Private Room' and 'Shared Room' contain the same information as applying pd.get_dummies() to 'Room Type' with drop_first = True and dtype = int parameters. Therefore, the 'Room Type' column can be removed.
abnb.drop('Room Type', axis = 1, inplace = True)

abnb['Person Capacity'].value_counts(dropna = False).sort_values(ascending = False)#.plot(kind = 'bar'); ### The feature has 5 possible values, with 2.0 and 4.0 accounting for over 70% of all observations. Interestingly, the average price for Person Capacity values of 2.0 and 3.0 is nearly identical, as is the case for values of 4.0 and 5.0. This suggests an opportunity for feature engineering by grouping the values into three bins: '2-3', '4-5', and '6', which may more accurately predict the flat price. Both features will be tested in the model (abnb.groupby('Person Capacity')['Price'].mean().plot(kind = 'bar');).
abnb['Person Capacity Gr'] = np.where(abnb['Person Capacity'] < 4, '2-3', np.where(abnb['Person Capacity'] > 5, '6', '4-5'))  ### A feature created this way, combined with the application of the pd.get_dummies() function, will likely serve as a strong price predictor (abnb.groupby(['Person Capacity Gr'])['Price'].mean().plot(kind = 'bar');).
abnb['Person Capacity Gr'] = abnb['Person Capacity Gr'].astype('category')

abnb['Superhost'].value_counts(dropna = False).sort_values()#.plot(kind = 'bar'); There are nearly three times more flats without the Superhost flag, and they are generally more expensive (258.6 vs. 236.8). The trend varies by city, making this feature likely less impactful for Linear Regression (abnb.groupby('Superhost')['Price'].mean().plot(kind = 'bar'); || abnb.groupby(['City', 'Superhost'])['Price'].mean().plot(kind = 'bar');). 

abnb.groupby(['Multiple Rooms'])['Price'].count()#.plot(kind = 'bar');  ### There are 2.4 times more flats without the 'Multiple Rooms' status, yet they are generally more expensive (261.2 vs. 231.9), a trend consistent across most cities (but not all eg. Berlin, Paris) and all Person Capacity bins. This feature will be tested in the model, though its impact is expected to be limited (abnb.groupby(['Multiple Rooms'])['Price'].mean().plot(kind = 'bar'); || abnb.groupby(['City', 'Multiple Rooms'])['Price'].mean().plot(kind='bar') || abnb.groupby(['Multiple Rooms', 'Person Capacity'])['Price'].mean()).

abnb.groupby(['Business'])['Price'].count()#.plot(kind = 'bar');  ### There are almost twice as many flats without the Business Flag as those with it. On average, flats with the Business Flag are slightly more expensive (257.5 vs. 250.0). In some cities, this difference is significant (e.g., Barcelona, Paris), while in others, it is small. Interestingly, in Amsterdam and Lisbon, flats without the Business Flag are, on average, more expensive (abnb.groupby(['Business'])['Price'].mean() || abnb.groupby(['City', 'Business'])['Price'].mean().plot(kind = 'bar');).

abnb.groupby(['Cleanliness Rating'])['Price'].count()#.plot(kind = 'bar'); ### Flats with a Cleanliness Rating of 8 to 10 account for nearly 97% of all observations, with average prices ranging between 250.8 and 253.6. Significant price differences are observed only for flats rated 5.0 or 3.0, which together account for only 60 observations. As a result, this feature is unlikely to be a strong predictor of flat rental prices (abnb.groupby('Cleanliness Rating')['Price'].mean().plot(kind = 'bar');).

abnb.groupby(['Guest Satisfaction'])['Price'].count()#.plot(kind = 'bar');  ### Only 1.6k observations represent flats with a Guest Satisfaction score below 80. Moreover, the average price does not strongly correlate with the Guest Satisfaction score. Flats with a perfect score of 100 (nearly 20% of all observations) have only a slightly higher average rental price than those scoring 80–99. Some lower scores (e.g., 57, 53, 65) show higher average prices than flats with a score of 100, but these are based on a limited number of observations. Therefore, this feature is unlikely to serve as an effective price predictor. However, a feature engineering approach, such as creating a flag for 100% satisfaction, will be tested (abnb.groupby(['Guest Satisfaction'])['Price'].mean().plot(kind = 'bar');).
abnb['100_GS'] = np.where(abnb['Guest Satisfaction'] == 100, 1, 0)

abnb.groupby(['Bedrooms'])['Price'].count()#.plot(kind = 'bar');  ### The data in this feature appears questionable regarding its accuracy. The author is not an expert in short-term flat bookings, but the data seems suspicious. One possible explanation could be that flat owners are not paying enough attention when filling out this field while posting an offer, or these strange values of 9 and 10 bedrooms could be a result of the web scraping phase. Of course, another explanation for this data discrepancy could also be true. About 9% of flats (3.7k observations) are listed as having 0 bedrooms, yet their average price is slightly higher than flats with 1 bedroom. There are also flats with 0 bedrooms but a Person Capacity of 2 to 6 people, as well as flats with, for example, 4 or 5 bedrooms and a Person Capacity of 2. Additionally, 78 observations include flats with 4 to 10 bedrooms, and interestingly, flats with 10 bedrooms have an average rental price up to 3 times lower than flats with 0 bedrooms (flats with 9 bedrooms also have surprisingly low average prices) || abnb.groupby(['Bedrooms'])['Price'].mean().plot(kind = 'bar');).
abnb['Bedrooms'] = abnb['Bedrooms'].astype('category')   ### The author has doubts regarding the values presented in the "Bedrooms" feature. However, the average prices for flats with different numbers of bedrooms vary significantly, which could make this feature a valuable price predictor. Therefore, this feature will be tested in the model, not using integer values, but as a categorical variable by importing it into the model using the pd.get_dummies() function.

print('Price vs City Center (km) corr:', round(abnb['City Center (km)'].corr(abnb['Price']),3), ' || Price vs Metro Distance (km) corr:',  round(abnb['Metro Distance (km)'].corr(abnb['Price']),3) ) 
#sns.scatterplot(data = abnb, x = 'City Center (km)', y = 'Price', s = 12.5, alpha = 0.5);
#sns.scatterplot(data = abnb, x = 'Metro Distance (km)', y = 'Price', s = 12.5, alpha = 0.5);  ### Two distance-related metrics, surprisingly, show minimal correlation with flat rental prices, contrary to the author's preferences when using platforms like Airbnb or Booking.com. Even more unexpectedly, the correlation in both cases is negative. Scatterplots reveal that the relationship between these metrics and price is non-linear. Consequently, the author plans to explore various feature engineering approaches, such as squaring the feature values, applying np.log(), binning, or even combining the two features into a single metric.

print('Price vs Attraction Index corr:', round(abnb['Attraction Index'].corr(abnb['Price']),3), ' || Price vs Restraunt Index corr:',  round(abnb['Restraunt Index'].corr(abnb['Price']),3) )
print('Price vs Normalised Attraction Index corr:', round(abnb['Normalised Attraction Index'].corr(abnb['Price']),3), ' || Price vs Normalised Restraunt Index corr:',  round(abnb['Normalised Restraunt Index'].corr(abnb['Price']),3) )
#sns.scatterplot(data = abnb, x = 'Attraction Index', y = 'Price', s = 12.5, alpha = 0.5); sns.scatterplot(data = abnb, x = 'Normalised Attraction Index', y = 'Price', s = 12.5, alpha = 0.5);
#sns.scatterplot(data = abnb, x = 'Restraunt Index', y = 'Price', s = 12.5, alpha = 0.5); sns.scatterplot(data = abnb, x = 'Normalised Restraunt Index', y = 'Price', s = 12.5, alpha = 0.5);  ### Four more numeric features are present in the dataframe, but their content is not precisely described on the Kaggle project website (https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset/data). The author suspects these features represent Attraction and Restaurant indices, which are no longer available on Airbnb (as of January 2025) but were calculated for each flat when the dataset was created (first Kaggle visible downloads: I 2024; first code post: IV 2023). Scatterplots show a non-linear but positive correlation with Price, with normalized indices versions having a stronger correlation.

abnb_test = abnb.copy() ### Creating a copy of the dataframe to generate new columns. Only columns that effectively predict Price and are valuable for the Linear Regression model will be added to the original dataframe.

abnb_test['AI'] = abnb_test['Attraction Index'] / abnb_test.groupby('City')['Attraction Index'].transform('max') * 100
abnb_test['AI2'] = abnb_test['Attraction Index'] / abnb_test.groupby('City')['Attraction Index'].transform(
    lambda x: x.nlargest(2).iloc[-1]) * 100

abnb_test['RI'] = abnb_test['Restraunt Index'] / abnb_test.groupby('City')['Restraunt Index'].transform('max') * 100
abnb_test['RI2'] = abnb_test['Restraunt Index'] / abnb_test.groupby('City')['Restraunt Index'].transform(
    lambda x: x.nlargest(2).iloc[-1]) * 100   ### The Kaggle project page does not explain how the normalized features were calculated. The author tried normalizing by dividing the raw value by the city's maximum, achieving 96.2–96.9% agreement for the Normalized Attraction Index (with a 0.1 threshold) but only 52.9–69% for the Normalized Restaurant Index. Reintroducing 124 removed Price outliers did not improve the match. Despite incomplete understanding, the author included these features in the model-building phase due to strong correlation values.


### USEFUL FUNCTIONS CREATION

def best_predictor_identifier(dataframe, dep_feature, test_size = 0.2, random_state = 999, importance_level = 0.005):

    df = pd.DataFrame({'Feature' : [], 'R2_Train' : [], 'R2_Test' : [], 'Numeric' : [], 'Corr' : [], 'Coef' : [], 'P_Coef' : [], 'F_Val' : [], 'P_F_Val' : []})
    dataframe_in = dataframe.copy()
    
    for column in dataframe_in.drop(dep_feature, axis = 1).columns:
        
        X = sm.add_constant(pd.get_dummies(dataframe_in.drop(dep_feature, axis = 1)[[column]], drop_first=True, dtype = int))
        y = dataframe_in[dep_feature]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
        model = sm.OLS(y_train, X_train).fit()
        
        r2_train = round(r2_score(y_train, model.predict(X_train)), 3)
        r2_test = round(r2_score(y_test, model.predict(X_test)), 3)
        
        numeric_type = pd.api.types.is_numeric_dtype(dataframe_in[column])
        if numeric_type:
            corr = dataframe_in[dep_feature].corr(dataframe_in[column])
            nr_type = 1
            coef = model.params.to_frame().reset_index().query("index != 'const'").iloc[:,1].mean()
            
        else:
            corr =  -2
            nr_type = 0
            coef = 0.000
 
        p_coef = model.pvalues.to_frame().reset_index().query("index != 'const'").iloc[:,1].mean()
        fval = model.fvalue
        pfval = model.f_pvalue

        df.loc[len(df)] = [column, r2_train, r2_test, nr_type, corr, coef, p_coef, fval, pfval]
        
    df = df[df['R2_Train'] >= importance_level]
    df = df.sort_values('R2_Train', ascending = False).reset_index(drop = True)
    
    return df

### best_predictor_identifier() function returns a dataframe with a list of independent features from the investigated dataframe and a set of calculated metrics per feature, which helps estimate the best features for a linear regression model. The function creates a copy of the original dataframe to ensure no information is lost during processing. In an iterative process, a temporary model is created that predicts the dependent feature value based on just one column. Useful metrics such as R2, correlation with the dependent value, and F-value are calculated using functions from the statsmodels and scikit-learn libraries.
### The final output is a dataframe filtered to include only features with sufficiently strong R2 values, assisting the user in identifying features to prioritize when manually constructing the model. The function is parameterized with inputs such as the dataframe, the dependent feature, the minimal R2 value, and additional options, making it adaptable for use in future projects.


def linear_model_builder(df, dep_feature, importance_level = 0.005, p_significance = 0.01, test_size = 0.2, random_state = 999): 
    
    R2_train, r2_impro, i  = -1.1, 1, 0
    df_copy = df.copy()
    product_df = pd.DataFrame({'Feature' : [], 'R2_Train' : [], 'R2_Test' : [], 'R2_Train improve' : [], 'R2_Test improve' : [], 'MAE' : [], 'MAPE' : [], 'Numeric' : [], 'F_Val' : [], 'P_F_Val' : [], 'High_P_Coef' : [], 'Last_Par_Coef' : [], 'Last_Par_P' : [], 'Corr' : [], 'Corr_vs_Coef' : [], 'DW' : []})
    features_list = []
    features_elimination = list(df_copy.columns)

    while r2_impro >= importance_level:
        for column in df_copy[features_elimination].drop(dep_feature, axis = 1).columns:
            
            X = sm.add_constant(pd.get_dummies(df_copy.drop(dep_feature, axis = 1)[features_list + [column]], drop_first=True, dtype = int))
            y = df_copy[dep_feature]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
            model = sm.OLS(y_train, X_train).fit()
            
            if round(r2_score(y_train, model.predict(X_train)), 3) > R2_train: 
                best_predictor = {}
                best_predictor[column] = round(r2_score(y_train, model.predict(X_train)), 3)
                R2_train = best_predictor[column]
                
                R2_test = round(r2_score(y_test, model.predict(X_test)), 3)
                
                fval = round(model.fvalue,3)
                pfval = round(model.f_pvalue,3)
                
                w = 0
                for p_value in model.pvalues:
                    if p_value > p_significance:
                        w += 1
                
                dw_val = round(sm.stats.stattools.durbin_watson(model.resid),3)
                        
                numeric_type = pd.api.types.is_numeric_dtype(df_copy[column])
                if numeric_type:
                    coefs = model.params.to_frame().reset_index()
                    p_values = model.pvalues.to_frame().reset_index()
                    last_param_coef = round(coefs[coefs['index'] == column].iloc[0,1],3)
                    last_param_pval = round(p_values[p_values['index'] == column].iloc[0,1],3)
                    
                    nr_type = 1
                    
                    corr = round(df_copy[dep_feature].corr(df_copy[column]),3)
                    
                    if (corr > 0 and last_param_coef > 0) or (corr < 0 and last_param_coef < 0):
                        corr_vs_coef = 1
                    else:
                        corr_vs_coef = 0 
     
                else:
                    last_param_coef = 0.000
                    p_values = model.pvalues.to_frame().reset_index()
                    last_param_pval = p_values[p_values['index'].str.contains(column)][0].mean()
                    
                    nr_type = 0
                    
                    corr = -2
                    corr_vs_coef = -1
                
                mae_val = round(mae(y_train, model.predict(X_train)),3)
                mape_val = round(mape(y_train, model.predict(X_train)),3)
                
        if i > 0:    
            r2_impro = round(R2_train - last_r2, 3)
            r2_impro_test = round(R2_test - last_r2_test, 3)
        else:
            r2_impro = R2_train
            r2_impro_test = R2_test
        
        features_list.append(list(best_predictor.keys())[-1])
        features_elimination.remove(list(best_predictor.keys())[-1])

        product_df.loc[len(product_df)] = [list(best_predictor.keys())[-1], R2_train, R2_test, r2_impro, r2_impro_test, mae_val, mape_val, nr_type, fval, pfval, w, last_param_coef, last_param_pval, corr, corr_vs_coef, dw_val]
        last_r2 = list(best_predictor.values())[-1]
        last_r2_test = R2_test
        i += 1
  
    return product_df

### linear_model_builder() is a function that iteratively builds a linear regression model. In each iteration, the function temporarily adds one new feature to the existing model, calculates the updated model's R2, and identifies the feature that yields the highest R2 improvement. This feature is then permanently added to the model and excluded from further iterations. The process repeats until no remaining feature can improve the R2 beyond the threshold specified in the function's parameters.
### As a final product, the function yields a dataframe with a list of features and model strength assessment metrics calculated after adding each feature to the model. Once a feature is selected and added to the model, the function calculates basic metrics using functions from the statsmodels and scikit-learn libraries. Metrics such as R2, Mean Absolute Error, Coefficient Value, and Durbin-Watson value are computed to help the user evaluate whether each feature's addition provides real value and improves the model's performance.

predictors_hierarchy_df = best_predictor_identifier(abnb_test,'Price')  ### The previously constructed function identified 'City' as the best predictor for 'Price.' Other promising metrics worth exploring in the feature engineering phase include Normalised Attraction Index, Normalised Restaurant Index, Bedrooms, and Person Capacity.
basic_linear_regr_model_df = linear_model_builder(abnb_test,'Price')  ### A linear regression model built using the previously constructed function achieved an R2 of 0.574, MAE of 74.14, and MAPE of 0.315. Interestingly, Person Capacity was ranked as the 2nd best feature to add to the model, despite being identified as the 9th best predictor when evaluated individually by the best_predictor_identifier() function. It is now time to proceed with feature engineering and assess whether the steps taken by the author can improve the current model further.


### FEATURE ENGINEERING

abnb_test['Bedrooms_x'] = abnb_test['Bedrooms'].astype(int)
columns_to_terms = ['Normalised Attraction Index', 'Normalised Restraunt Index', 'City Center (km)', 'Metro Distance (km)', 'Person Capacity', 'Bedrooms_x']
for column in columns_to_terms:
    abnb_test[column + '_2'] = abnb_test[column] ** 2
    abnb_test[column + '_3'] = abnb_test[column] ** 3
 
columns_to_log = ['Normalised Attraction Index', 'Normalised Restraunt Index', 'City Center (km)', 'Metro Distance (km)', 'Person Capacity']
for column in columns_to_log:
    abnb_test[column + '_L'] = np.log(abnb_test[column]) ### This part of the code performs common feature engineering operations on numeric columns. In an iterative process, a few selected columns are raised to the 2nd and 3rd powers and transformed using np.log() to create new columns. These newly created columns will be evaluated for their predictive power in the model.

round(abnb_test['Person Capacity'].corr(abnb_test['Bedrooms']),3)
abnb_test['Bedrooms GR'] = np.where(abnb_test['Bedrooms_x'] >= 9, 1, np.where(abnb_test['Bedrooms_x'] < 2, 2, abnb_test['Bedrooms_x'] + 1) )
abnb_test['Bedrooms GRS'] = abnb_test['Bedrooms GR'].astype(str)
abnb_test['Person vs Bedrooms'] = abnb_test['Person Capacity'] / abnb_test['Bedrooms GR'] 
abnb_test['Bedrooms vs Person'] = abnb_test['Bedrooms GR'] / abnb_test['Person Capacity']  ### Person Capacity and Bedrooms are two independent features with a strong correlation (0.558), which could lead to multicollinearity issues. To address this, the author combined these features by dividing one by the other (and vice versa). Spoiler alert: as we will see, none of these derived metrics outperformed the original, unmodified Bedroom feature as a Price predictor.


def cc_dist_bin_creator(df = abnb_test, dep_feature = 'Price', reg_column = 'City Center (km)', new_column_name = 'CC (km) GR2', bins = [0,1]):
    labels = []
    alphabet = [chr(i) for i in range(65, 91)]
    a = 0
    
    if len(bins) > 53:
        return 'According to how the function was built, the maximum length of bins is 50. Please reduce the number of bins being used in the function and call it again.'
    
    for x in range(len(bins)):
        if x < len(bins) - 2:
            if a <= 25:
                labels.append(alphabet[a] + '. ' + str(bins[x]) + '-' + str(bins[x+1]))
            elif a < 52:
                labels.append('Z' + alphabet[a-26] + '. ' + str(bins[x]) + '-' + str(bins[x+1]))
        else:
            if a <= 25:
                labels.append(alphabet[a] + '. ' + str(bins[x]) + '+')
                break
            elif a < 52:
                labels.append('Z' + alphabet[a-26] + '. ' + str(bins[x]) + '+')
                break
        a += 1
    
    df[new_column_name] = pd.cut(df[reg_column], bins = bins, labels = labels)
    df[new_column_name] = df[new_column_name].astype('string')
    
    temp_df = pd.DataFrame( { dep_feature : df.groupby(new_column_name)[dep_feature].mean() , 'CNT' : df.groupby(new_column_name)[dep_feature].count(), 'CNT_Pct' : df.groupby(new_column_name)[dep_feature].count() / len(df) * 100} ).sort_index()
    return temp_df

### The function categorizes values from an existing column into user-defined bins, modifying the pd.cut() function to include automatic label creation based on the bins. It also returns a temporary dataframe with sorted values, enabling assessment of whether the created bins differ in the average value of the metric predicted in the linear regression model.
max_distance = [abnb_test['City Center (km)'].max()]

bins = list(np.linspace(0,15,4)) + max_distance
cc_dist_bin_creator(new_column_name = 'CC (km) GR', bins = bins)  ### The average flat rental price decreases as the distance from the city center increases. Only 84 observations exceed 15 km from the city center, while 90% of observations fall within the 0–5 km range, warranting a more detailed investigation of this cluster.

bins = list(np.linspace(0,10,5)) + max_distance
cc_dist_bin_creator(new_column_name = 'CC (km) GR2', bins = bins) ### 55.8% of observations are in the 0–2.5 km bin, and 34.1% in the 2.5–5 km bin, suggesting potential for finer grouping. Average prices in the 5–7.5 km and 7.5–10 km ranges are similar, with only 380 observations beyond 10 km.

bins = list(range(0, 9, 1)) + max_distance
cc_dist_bin_creator(new_column_name = 'CC (km) GR3', bins = bins) ### Flats within 1 km of the City Center have the highest rental prices. The 1–2, 2–3, and 3–4 km clusters show similar average prices, making up about two-thirds of all observations. Flats 6 km or farther are the cheapest to rent on average, representing around 5.5% of all observations.

bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 9] + max_distance
cc_dist_bin_creator(new_column_name = 'CC (km) GR4', bins = bins) ### Flats closest to the City Center (0-0.5 km and 0.5-1 km) differ in price from others but are similar to each other. Prices from 1.5-2.0 km to 3.5-4.0 km show no linear trend and are roughly similar. Additional clusters could be 4.0-6.0 km, 6.0-9.0 km, and beyond 9 km.

bins = list(np.linspace(0,1.5,7)) + [4, 6, 9] + max_distance
cc_dist_bin_creator(new_column_name = 'CC (km) GR5', bins = bins) ### Rental prices for flats within 0.25 km of the City Center are the highest. Prices for bins 0.25–0.5 to 1.0–1.25 km are similar and could be grouped. A revised binning structure could include 1.25–4 km, along with the existing bins: 4–6 km, 6–9 km, and 9+ km.

bins = [0, 0.25, 1.5, 4, 6, 9] + max_distance
cc_dist_bin_creator(new_column_name = 'CC (km) GR6', bins = bins) ### Rental prices decrease with distance from the City Center. The highest-priced group (0–0.25 km) accounts for just 1.25% of observations, while the 1.5–4 km bin includes nearly half. Previous inspections revealed no distinct price trends within these ranges, suggesting this categorization is unlikely to outperform the original City Center (km) feature in prediction.

predictors_hierarchy_df_later = best_predictor_identifier(abnb_test,'Price')  ### Among the top 10 features identified as the best solo predictors of the dependent variable, 8 were engineered by the author. However, their R2 values are only slightly better than those of the original dataframe features they were based on, offering limited potential to significantly enhance the model's predictive power. 
basic_linear_regr_model_df_later = linear_model_builder(abnb_test,'Price')  ### As predicted, only two features engineered by the author were included in the model. Notably, these features ranked as the 6th and 7th choices, indicating they are not significant game changers. 

abnb['Normalised Attraction Index_2'] = abnb['Normalised Attraction Index'] ** 2
model_features = list(basic_linear_regr_model_df_later['Feature']) + ['Price']


### LINEAR REGRESSION

def linear_regression_edited(df, dep_feature, test_size = 0.2, random_state = 999):

    X = sm.add_constant(pd.get_dummies(df.drop(dep_feature, axis = 1), drop_first=True, dtype = int))
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    model = sm.OLS(y_train, X_train).fit()

    print()
    print()
    print('+++++ Linear Regression Model summary using a dataframe with ' + str(len(df.columns)) + ' columns to predict ' + dep_feature + ' +++++')
    
    print()
    print('MAE TRAIN:', round(mae(y_train, model.predict(X_train)), 3), ' || MAE TEST:', round(mae(y_test, model.predict(X_test)), 3))
    print('MAPE TRAIN:', round(mape(y_train, model.predict(X_train)) * 100, 3), ' || MAPE TEST:', round(mape(y_test, model.predict(X_test)) * 100, 3))
    print('R2 TRAIN:', round(r2_score(y_train, model.predict(X_train)), 3), ' || R2 TEST:', round(r2_score(y_test, model.predict(X_test)), 3))
    print()
    print('MODEL PROB (F-statistic):', round(model.f_pvalue, 3))
    print('DURBIN-WATSON:', round(sm.stats.stattools.durbin_watson(model.resid),3))

    k = 0
    for var, p in model.pvalues.items():
        if p > 0.01:
            print(f"--- {var}: {round(p,3)}")
            k += 1
            if k >= 5:
                print('There are/is ' + str(len(model.pvalues) - k) + ' more coefficient/s with p-values higher than the set threshold. The first 5 were listed above.')
                break
    if k == 0:
        print("COEF: +++ The p-values for all model coefficients are below 0.01, indicating statistical significance at the 0.01 level.")

### A function designed to generate a Linear Regression model from a given dataframe and dependent feature. It splits the data into training and test sets using train_test_split() and returns metrics such as R2, MAE, MAPE, and additional performance measures for both training and test datasets.

abnb_filtered = abnb.drop('Normalised Attraction Index_2', axis = 1).copy()
abnb_test_filtered = abnb_test.copy()

abnb_filtered['Price_ix'] = abnb_filtered['Price'] / abnb_filtered.groupby('City')['Price'].transform('mean')
abnb_test_filtered['Price_ix'] = abnb_test_filtered['Price'] / abnb_test_filtered.groupby('City')['Price'].transform('mean')

abnb_filtered = abnb_filtered[abnb_filtered['Price_ix'] <= 3].drop({'Price_ix'}, axis = 1)  
abnb_test_filtered = abnb_test_filtered[abnb_test_filtered['Price_ix'] <= 3].drop({'Price_ix'}, axis = 1) 

basic_linear_regr_model_df_later_filtered = linear_model_builder(abnb_test_filtered, 'Price')
model_features_filtered = list(basic_linear_regr_model_df_later_filtered['Feature']) + ['Price']
abnb_filtered['Normalised Attraction Index_2'] = abnb_filtered['Normalised Attraction Index'] ** 2   ### The code creates copies of previously structured dataframes and replicates the method used to create the Price_ix function. This time, it filters out flats with rental prices three times or more above the average in their city. The author is curious to see how this operation will impact model parameters.

print()
print('Models that base on the data, where flats, which rental price is higher than 4.5 average rental price in that city are excluded')
linear_regression_edited(abnb[model_features], 'Price')                     ### R2: 0.577 ||| MAPE: 31.612 ||| MAE: 74.141 ---> achieved from a dataframe with 9 independent columns
linear_regression_edited(abnb, 'Price')                                     ### R2: 0.581 ||| MAPE: 31.419 ||| MAE: 73.778 ---> achieved from a dataframe with 20 independent columns 
linear_regression_edited(abnb_test, 'Price')                                ### R2: 0.589 ||| MAPE: 31.158 ||| MAE: 73.164 ---> achieved from a dataframe with 51 independent columns. Model accuracy is improving, but the slight changes don't justify extending the dataframe with so many additional columns.

print()
print('Models that base on the data, where flats, which rental price is higher than 3.0 average rental price in that city are excluded')
linear_regression_edited(abnb_filtered[model_features_filtered], 'Price')   ### R2: 0.614 ||| MAPE: 29.436 ||| MAE: 67.255
linear_regression_edited(abnb_filtered, 'Price')                            ### R2: 0.618 ||| MAPE: 29.390 ||| MAE: 67.045
linear_regression_edited(abnb_test_filtered, 'Price')                       ### R2: 0.625 ||| MAPE: 29.147 ||| MAE: 66.466 ---> It is evident that a stricter approach to outlier exclusion improved the model's predictive power, as verified on the test dataset. R2 increased by approximately 0.37, while MAE and MAPE decreased by around 2 points and 6.7 points, respectively.


