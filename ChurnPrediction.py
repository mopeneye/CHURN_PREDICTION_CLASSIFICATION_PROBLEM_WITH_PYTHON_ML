# Problem:
# Can you develop a machine learning model that can predict customers who will leave the company?
#
# The aim is to predict whether a bank's customers leave the bank or not.
#
# The event that defines customer abandonment is the customer closing his bank account.
#
# Data Set Story:
#
# It consists of 10000 observations and 12 variables.
# Independent variables contain information about customers.
# Dependent variable expresses customer abandonment status.
# Variables:
#
# Surname: Last name
# CreditScore: Credit score
# Geography: Country (Germany / France / Spain)
# Gender: Gender (Female / Male)
# Age: Age
# Tenure: How many years of customers
# Balance: Balance
# NumOfProducts: Bank product used
# HasCrCard: Credit card status (0 = No, 1 = Yes)
# IsActiveMember: Active membership status (0 = No, 1 = Yes)
# EstimatedSalary: Estimated salary
# Exited: Abandoned or not? (0 = No, 1 = Yes)

# Data Understanding

# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None)

def load_churn_data():
    df = pd.read_csv(r'E:\PROJECTS\dsmlbc\ChurnPrediction\datasets\churn.csv', index_col=0)
    return df

df = load_churn_data()

# EDA

# OVERVIEW

print(df.head())

print(df.tail())

print(df.info())

print(df.columns)

print(df.shape)

print(df.index)

print(df.describe().T)

print(df.isnull().values.any())

print(df.isnull().sum().sort_values(ascending=False))

# INDEPENDENT VARIABLE OPERATIONS

df['HasCrCard'].value_counts() #should be converted to category

df["HasCrCard"] = df["HasCrCard"].astype('O')

df['IsActiveMember'].value_counts() #should be converted to category

df["IsActiveMember"] = df["IsActiveMember"].astype('O')

df["NumOfProducts"] = df["NumOfProducts"].astype('O')

df.loc[(df['Tenure'] == 0), 'Tenure'] = 0.5

df['Tenure'].value_counts()

category = pd.cut(df.Tenure,bins=[0,1,4,7,12],labels=['New_customers','Important_customers','VeryImportant_customers','ExtremelyImportant_customers'])

df.insert(6, 'Tenure_category', category)

df.drop(['CustomerId', 'Surname', 'Tenure'], axis=1, inplace=True)

df['Tenure_category'] = df['Tenure_category'].astype('O')

# 2. CATEGORICAL VARIABLE ANALYSIS

# WHAT ARE THE NAMES OF CATEGORICAL VARIABLES?
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Categorical Variable count: ', len(cat_cols))
print(cat_cols)

# HOW MANY CLASSES DO CATEGORICAL VARIABLES HAVE?

print(df[cat_cols].nunique())

def cats_summary(data, categorical_cols, number_of_classes=10):
    var_count = 0  # count of categorical variables will be reported
    vars_more_classes = []  # categorical variables that have more than a number specified.
    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # choose according to class count
                print(pd.DataFrame({var: data[var].value_counts(), "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
                sns.countplot(x=var, data=data)
                plt.show()
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)

cats_summary(df, cat_cols)

# NUMERICAL VARIABLE ANALYSIS

print(df.describe().T)

# NUMERICAL VARIABLES COUNT OF DATASET?

num_cols = [col for col in df.columns if df[col].dtypes != 'O']
print('Numerical Variables Count: ', len(num_cols))
print('Numerical Variables: ', num_cols)

# Histograms for numerical variables?

def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")

hist_for_nums(df, num_cols)

# DISTRIBUTION OF "EXITED" VARIABLE

print(df["Exited"].value_counts()) #inbalancing problem!

def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target);
    facet.add_legend()

# TARGET ANALYSIS BASED ON CATEGORICAL VARIABLES

def target_summary_with_cat(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]
    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")
        plot_categories(df, cat=var, target='Exited')
        plt.show()

target_summary_with_cat(df, "Exited")

# TARGET ANALYSIS BASED ON NUMERICAL VARIABLES

def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")

target_summary_with_nums(df, "Exited")

# INVESTIGATION OF NUMERICAL VARIABLES EACH OTHER

def correlation_matrix(df):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show()

correlation_matrix(df)

# 6. WORK WITH OUTLIERS

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 0.5 * interquantile_range
    return low_limit, up_limit


num_cols2 = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].unique()) > 10]

def Has_outliers(data, number_col_names, plot=False):
    Outlier_variable_list = []

    for col in number_col_names:
        low, high = outlier_thresholds(df, col)

        if (df[(data[col] < low) | (data[col] > high)].shape[0] > 0):
            Outlier_variable_list.append(col)
            if (plot == True):
                sns.boxplot(x=col, data=df)
                plt.show()
    print('Variables that has outliers: ', Outlier_variable_list)
    return Outlier_variable_list


def Replace_with_thresholds(data, col):
    low, up = outlier_thresholds(data, col)
    data.loc[(data[col] < low), col] = low
    data.loc[(data[col] > up), col] = up
    print("Outliers for ", col, "column have been replaced with thresholds ",
          low, " and ", up)


var_names = Has_outliers(df, num_cols2, True)

# print(var_names)

for col in var_names:
    Replace_with_thresholds(df, col)

Has_outliers(df, num_cols2, True)


# MISSING VALUE ANALYSIS

# Is there any missing values
print(df.isnull().values.any()) #NO!

# 8. LABEL ENCODING

def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder()

    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                  and len(dataframe[col].value_counts()) == 2]

    for col in label_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe

#
df = label_encoder(df)

# ONE-HOT ENCODING
def one_hot_encoder(dataframe, category_freq=20, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O']

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    return dataframe


df = one_hot_encoder(df)

# MODELLING

y = df["Exited"]
X = df.drop(["Exited"], axis=1)


results = []
names = []

models = [('RF', RandomForestClassifier())]
     # ('XGB', GradientBoostingClassifier())]
    #      ("LightGBM", LGBMClassifier())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print('Base: ', msg)

    # RF Tuned
    if name == 'RF':
        rf_params = {"n_estimators": [1500, 3000],
                     "max_features": [5, 10],
                     "min_samples_split": [10, 20],
                     "max_depth": [50, None]}

        rf_model = RandomForestClassifier(random_state=123)
        print('RF Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(rf_model,
                             rf_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, verbose=2, n_jobs=-1).fit(X, y)  # ???
        print('RF Bitis zamani: ', datetime.now())
        rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(rf_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('RF Tuned: ', msg)
        print('RF Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(rf_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('rf_importances.png')

    # LGBM Tuned
    elif name == 'LightGBM':
        lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
        "n_estimators": [500, 1000, 1500, 3000, 5000],
        "max_depth": [3, 5, 8, 10, 20, 50],
        'num_leaves': [31, 50, 100]}

        lgbm_model = LGBMClassifier(random_state=123)
        print('LGBM Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(lgbm_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('LGBM Bitis zamani: ', datetime.now())
        lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(lgbm_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('LGBM Tuned: ', msg)
        print('LGBM Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('lgbm_importances.png')

    # XGB Tuned
    elif name == 'XGB':
        xgb_params = {#"colsample_bytree": [0.05, 0.1, 0.5, 1],
                      'max_depth': np.arange(1, 11),
                      'subsample': [0.5, 1, 5],
                      'learning_rate': [0.005, 0.01],
                      'n_estimators': [100, 500, 1000],
                      'loss': ['deviance', 'exponential']}

        xgb_model = GradientBoostingClassifier(random_state=123)

        print('XGB Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(xgb_model,
                             xgb_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('XGB Bitis zamani: ', datetime.now())
        xgb_tuned = GradientBoostingClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(xgb_tuned, X, y, cv=10,        scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('XGB Tuned: ', msg)
        print('XGB Best params: ', gs_cv.best_params_)


# LGBM
# Base:  LightGBM: 0.860300 (0.009890)
# LGBM Baslangic zamani:  2020-11-07 19:09:06.606067
# Fitting 10 folds for each of 270 candidates, totalling 2700 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# LGBM Bitis zamani:  2020-11-07 19:56:19.968349
# LGBM Tuned:  LightGBM: 0.865300 (0.000000)
# LGBM Best params:  {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 3000, 'num_leaves': 31}
#
# XGB
# Base:  XGB: 0.863100 (0.008882)
# XGB Baslangic zamani:  2020-11-07 21:17:24.579926
# Fitting 10 folds for each of 360 candidates, totalling 3600 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# XGB Bitis zamani:  2020-11-07 23:19:54.800823
# XGB Tuned:  XGB: 0.864900 (0.000000)
# XGB Best params:  {'learning_rate': 0.005, 'loss': 'deviance', 'max_depth': 8, 'n_estimators': 500, 'subsample': 0.5}
#
# RF
# Base:  RF: 0.861800 (0.009357)
# RF Baslangic zamani:  2020-11-08 10:42:06.847971
# Fitting 10 folds for each of 108 candidates, totalling 1080 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# RF Bitis zamani:  2020-11-08 19:15:02.524045
# RF Tuned:  RF: 0.866700 (0.000000)
# RF Best params:  {'max_depth': 50, 'max_features': 5, 'min_samples_split': 10, 'n_estimators': 1500}







