import pandas as pd 
import numpy as np

import random

import category_encoders as ce
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
import bisect

from sklearn.preprocessing import OneHotEncoder

import shap

from hyperopt import fmin, tpe, hp, anneal, Trials

import os
import gc

import datetime

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def process_data(path,drop_columns):
    df=pd.read_csv(path).drop(columns=(drop_columns))
    return df

def understand_data(df,id_cols):
    print(f"Numerical Cloumns : {list(df.drop(columns=id_cols).select_dtypes(include='number'))}")
    print('-'*30)
    print(f"Categorical Cloumns : {list(df.drop(columns=id_cols).select_dtypes(include='O'))}")
    print('-'*30)
    print("Null values count :" )
    print()
    print(df.drop(columns=id_cols).isna().sum())
    print('-'*30)
    print()
    print(f"Zero  Variance Columns : {df.drop(columns=id_cols).select_dtypes(include='number').loc[:,df.drop(columns=id_cols).select_dtypes(include='number').std()==0].columns}")

def data_split(df):
    train=df[df.yearmo<=202203]
    val =df[df.yearmo==202204]
    hold_out=df[df.yearmo==202205]
    return train.reset_index(drop=True),val.reset_index(drop=True),hold_out.reset_index(drop=True)

def dpd_roll_rate(df):
    
    data={}
    for dpd in [0,30,60,90]:
        user_count=len(df[df['max_dpd']>=dpd])  
        data[dpd]=user_count
    dpd_flow=pd.DataFrame(list(data.items()),columns=['dpd',"user_count"])
    dpd_flow['user_percent']=round(dpd_flow['user_count']*100/max(dpd_flow['user_count']),2)

    return dpd_flow

def window_roll_rate(df,dpd):
    df2=df[df['max_dpd']>=dpd]
    df2['first_default']=np.where(df2['emi_1_dpd']>=dpd,1,
                              np.where(df2['emi_2_dpd']>=dpd,2,
                                       np.where(df2['emi_3_dpd']>=dpd,3,
                                                np.where(df2['emi_4_dpd']>=dpd,4,
                                                         np.where(df2['emi_5_dpd']>=dpd,5,
                                                                  np.where(df2['emi_6_dpd']>=dpd,6,0))))))

    window_roll=df2.groupby('first_default')['User_id'].count().reset_index()
    window_roll['user_percent']=round(window_roll['User_id']*100/sum(window_roll['User_id']),2)
    return window_roll

def create_label(df,dpd,months):
    months=["emi_"+str(x)+"_dpd" for x in range(1,months+1)]
    df['label']=np.where(df[months].max(axis=1)>=dpd,1,0)
    print("label column added to dataframe")
    return df

def label_distribution(dataset_list,dataset_list_name,label_name):
    i=0
    for d in dataset_list:
        label_distribution = pd.DataFrame(d[label_name].value_counts()).reset_index()
        label_distribution.columns=['label','user_count']
        label_distribution["%_users"]=label_distribution['user_count']*100/sum(label_distribution['user_count'])
        print('-'*30)
        print(f"label distribution of {dataset_list_name[i]}")
        print(label_distribution)
        i+=1


    
def derived_features(df):
    #replace([np.inf, -np.inf] due to division by zero, they are replaced with 0
    #fillna(0) If there are any missing values (NaN), they are filled with 0
    df['interest_received_ratio']=(df['interest_received']/df['total_payement']).replace([np.inf, -np.inf], 0).fillna(0)
    df['total_payement_per_loan'] = (df['total_payement']/df['number_of_loans']).replace([np.inf, -np.inf], 0).fillna(0)
    df['delinq_2yrs_ratio'] = (df['delinq_2yrs']/df['number_of_loans']).replace([np.inf, -np.inf], 0).fillna(0)

    return df

class eda:
    """EDA Class
    - Univarite EDA
      - Numeric Features Summary
      - Categorical Features Summary
    - Bivariate EDA
      - Correlation Plot
      - Box Plot"""
    def __init__(self,df,id_cols):
        self.df=df
        self.id_cols=id_cols
        self.num_cols=df.drop(columns=id_cols).select_dtypes(include='number').columns
        self.cat_cols=df.drop(columns=id_cols).select_dtypes(include='O').columns
    
    def numeric_summary(self):
        return self.df[self.num_cols].describe().T
    
    def categorical_summary(self):
        return self.df[self.cat_cols].describe().T
    
    def correlation_plot(self):
        """Correlation Plot of Numerical Features"""
        corr = self.df[self.num_cols].corr()

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True

        plt.figure(figsize=(len(corr.columns), len(corr.columns)))
        with sns.axes_style("white"):
            ax = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                             square=True, annot=True)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_autoscalex_on(True)
        ax.set_autoscaley_on(True)

        plt.show()

    def box_plot(self, group):
        """Box Plot of Numerical Features vs Group Features
        Parameters
        ----------
        group : List of Features (against with box plot of numerica features to be done)
        """
        for g in group:
            for col in self.num_cols:
                sns.boxplot(x = g, y = col, data = self.df)
                plt.ylabel('Values')
                plt.title(col)
                plt.show()

class categorical_encoding:
    """Target Encoding of categorical variables
    input dataframe, categorical columns, label name, parameters of target_encoder
    """
    def __init__(self,params):
        """
        Parameters
        ----------
        params : Dict
        """
        self.params = params

    def fit(self, df, cat_cols, label):
        """Fitting Encoder
        Parameters
        ----------
        df : DataFrame
        cat_cols : List (Categorical columns)
        label : String
        """
        self.te = ce.target_encoder.TargetEncoder(**self.params)
        self.te.fit(df[cat_cols], df[label])
    
    def transform(self, d):
        """Transforming Data Encode and inplace transform categorical features
        Parameters
        ----------
        d : DataFrame
        
        Returns
        -------
        d : DataFrame
        """
        d = pd.concat([d.drop(columns = self.te.feature_names), self.te.transform(d[self.te.feature_names])], axis = 1)
        return d
    
def random_forest_zero_importance(df,id_cols,label,params):
    rf=RandomForestClassifier(**params)
    rf.fit(df.drop(columns=id_cols).fillna(0),df[label])
    fi=pd.DataFrame({"features":df.drop(columns=id_cols).columns,"importance":rf.feature_importances_})
    zero_fi=fi[fi['importance']==0]['features']

    return zero_fi

def decision_tree_zero_importance(df,id_cols,label,params):
    rf=DecisionTreeClassifier(**params)
    rf.fit(df.drop(columns=id_cols).fillna(0),df[label])
    fi=pd.DataFrame({"features":df.drop(columns=id_cols).columns,"importance":rf.feature_importances_})
    zero_fi=fi[fi['importance']==0]['features']

    return zero_fi
        
    
def roc_auc(target_list, pred_list):
    """Print ROC AUC for Target and Predictions
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    """
    
    print(roc_auc_score(target_list[0], pred_list[0]))
    print(roc_auc_score(target_list[1], pred_list[1]))
    print(roc_auc_score(target_list[2], pred_list[2]))

def pr_auc(target_list, pred_list):
    """Print PR AUC Values
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    """
    
    pr, re, thresholds = precision_recall_curve(target_list[0], pred_list[0])
    pr_val, re_val, thresholds_val = precision_recall_curve(target_list[1], pred_list[1])
    pr_hold_out, re_hold_out, thresholds_hold_out = precision_recall_curve(target_list[2], pred_list[2])
    print(auc(re, pr))
    print(auc(re_val, pr_val))
    print(auc(re_hold_out, pr_hold_out))

def roc_auc_curve(target_list, pred_list):
    """Print ROC AUC Curve
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    """
    
    fpr, tpr, thresholds = roc_curve(target_list[0], pred_list[0])
    fpr_val, tpr_val, thresholds_test = roc_curve(target_list[1], pred_list[1])
    fpr_hold_out, tpr_hold_out, thresholds_hold_out = roc_curve(target_list[2], pred_list[2])

    
    roc_auc = roc_auc_score(target_list[0], pred_list[0])
    roc_auc_val = roc_auc_score(target_list[1], pred_list[1])
    roc_auc_hold_out = roc_auc_score(target_list[2], pred_list[2])


    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'Train AUC = %0.3f' % roc_auc, color = 'C0')
    plt.plot(fpr_val, tpr_val, 'b', label = 'Val AUC = %0.3f' % roc_auc_val, color = 'C1')
    plt.plot(fpr_hold_out, tpr_hold_out, 'b', label = 'Hold Out AUC = %0.3f' % roc_auc_hold_out, color = 'C2')
    #plt.plot(fpr_true, tpr_true, 'b', label = 'True Values AUC = %0.3f' % roc_auc_oot, color = 'C3')

    plt.legend(loc='best')
    plt.plot([0, 1], [0, 1],'r--', color = 'black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def pr_auc_curve(target_list, pred_list):
    """Print PR AUC Curve
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    """

    pr, re, thresholds = precision_recall_curve(target_list[0], pred_list[0])
    pr_val, re_val, thresholds_val = precision_recall_curve(target_list[1], pred_list[1])
    pr_hold_out, re_hold_out, thresholds_hold_out = precision_recall_curve(target_list[2], pred_list[2])
    
    precision_score_train = average_precision_score(target_list[0], pred_list[0])
    precision_score_val = average_precision_score(target_list[1], pred_list[1])
    precision_score_hold_out = average_precision_score(target_list[2], pred_list[2])
    
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.title('Precision Recall Curve')
    plt.plot(re, pr, 'b', label = 'Train Precision = %0.3f' % precision_score_train, color = 'C0')
    plt.plot(re_val, pr_val,  'b', label = 'Val Precision = %0.3f' % precision_score_val, color = 'C1')
    plt.plot(re_hold_out, pr_hold_out,  'b', label = 'Hold Out Precision = %0.3f' % precision_score_hold_out, color = 'C2')
    plt.legend(loc='best')
    #plt.plot([0, 1], [0, 1],'r--', color = 'black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

def score_distribution(target_list, pred_list, data_type_list):
    """Print Score Distribution Plots
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list
        Data Tagging like Train, Val, Hold Out
    """

    for i in range(len(data_type_list)):
        y_actual = target_list[i]
        y_predicted = pred_list[i]
        df_type = data_type_list[i]
        sub_df = pd.DataFrame({"y_actual": y_actual, "y_predicted": y_predicted})
        
        f, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=(16, 6))
        sns.distplot(sub_df[sub_df['y_actual']==1].y_predicted.values, hist=True, kde=True, rug=False, label="Defaulter", ax=ax)
        sns.distplot(sub_df[sub_df['y_actual']==0].y_predicted.values, hist=True, kde=True, rug=False, label="Non-Defaulter", ax=ax)
        plt.xlabel('Predicted positive class score')
        plt.ylabel('Count')
        plt.title(str(df_type) +' Distribution of predicted score')
        plt.legend(loc="upper right")
        plt.show()

def shap_importance(model, data_list, data_type_list):
    """Plot SHAP for top 20 features
    Parameters
    ----------
    model : object
        Model Object (Classifier).
    data_list : list
        List of Multiple DataFrames
    data_type_list : list
        Data Tagging like Train, Val, Hold Out
    """
    explainer = shap.TreeExplainer(model)
    i = 0
    for d in data_list:
        tmp_shap_values = explainer.shap_values(d[model.feature_name()])

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(14.5, 10.5))
        
        shap.summary_plot(tmp_shap_values[0], d[model.feature_name()], plot_type="dot", max_display=20, show=False, ax=ax)
        
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="x", labelsize=15)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.set_title(data_type_list[i] + " Shap Values")
        ax.set(ylabel="Features", xlabel="Mean SHAP Value")
        
        plt.show()
        i = i + 1

def class_rate(target_list, pred_list, data_type_list):

    """Print Class Rate Curves
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list
        Data Tagging like Train, Val, Hold Out
    """
    def buckets(y_actual, y_predicted, bins):
        df = pd.DataFrame({"y_actual": y_actual, "y_predicted": y_predicted})
        if bins is None:
            out, bins = pd.qcut(y_predicted, 30, retbins=True)
            df['score_bucket'] = pd.cut(df["y_predicted"], bins=bins)#, labels=range(20))
            #df['score_bins'] = bins[0:20]
            return df, bins
        else:
            df['score_bucket'] = pd.cut(df["y_predicted"], bins=bins)#, labels=range(20))
            return df

    def slope_df(actual, predicted, data_type):
        slope = pd.DataFrame(columns=['score_bucket', 'score_bins','count', 'sum', 'positive_class_rate', 'volume_percentage','Data'])
        for i in range(len(data_type)):
            y_actual = actual[i]
            y_predicted = predicted[i]
            df_type = data_type[i]
            if df_type == "Train":
                df_bucket, bins = buckets(y_actual, y_predicted, None)
            else:
                df_bucket = buckets(y_actual, y_predicted, bins)
            df_slope = df_bucket.groupby(['score_bucket'])["y_actual"].agg(['count', 'sum']).sort_index(ascending=False).reset_index()
            df_slope['positive_class_rate'] = (df_slope['sum'] / df_slope['count'])
            df_slope['volume_percentage'] = df_slope['count'] / df_slope['count'].sum()
            df_slope['Data'] = df_type
            slope = pd.concat([df_slope, slope], ignore_index=True)
        slope = slope.reset_index(drop = True)
        return slope, bins

    def slope_plot(df):
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        
        ax1 = sns.pointplot(x="score_bucket", y="positive_class_rate", data=df, hue="Data")
        ax1.set(ylabel="% Default", xlabel="Score Buckets")
        ax1.legend(loc='center right')
        ax1.set_xticklabels(df["score_bucket"].unique().tolist(), rotation=90)
        ax1.set_title("Bucket wise % Default")
        
        ax2 = ax1.twinx()
        ax2 = sns.barplot(x="score_bucket", y="volume_percentage", hue="Data", data=df, **{'alpha': 0.3})
        ax2.set(ylabel="Percentage of Volume", xlabel="")
        ax2.legend(loc='upper left')
        
        plt.show()

    slope, slope_bins = slope_df(target_list, pred_list, data_type_list)
    slope_plot(slope)

def cutoff_score(label, prediction, default_rate): 
    """Cutoff Score at a particular default rate 
    Parameters
    ----------
    label : Array
        Labels according to which cutoff need to be decided.
    prediction : Array
        Model Scores
    default_rate : Float
        Desired Cummulative default rate
    """
    pred = pd.DataFrame({"label":label, "score":prediction}).sort_values(by = 'score').reset_index(drop = True)
    pred['cummulative_defaulters'] = pred['label'].cumsum(axis = 0)
    pred['cummulative_default'] = (pred['cummulative_defaulters']/pred.index).fillna(0)
    cutoff = pred[pred.cummulative_default<=default_rate]['score'].max()
    
    return cutoff

    





















