#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:48:28 2019

@author: ubuntu
"""

#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# #Importing Libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn import preprocessing


# =============================================================================
#  Reading data
# =============================================================================
sales_train = pd.read_csv("sales_train.csv")
items = pd.read_csv("items.csv")
item_categories = pd.read_csv("item_categories.csv") 
shops = pd.read_csv("shops.csv")

# =============================================================================
# Merging data
# =============================================================================
sales_train_CC_items = sales_train.merge(items, 
                                         on="item_id")

sales_train_CC_items_CC_item_categories = sales_train_CC_items.merge(item_categories, 
                                                                     on="item_category_id")
sales_train_CC_items_CC_item_categories_CC_shops = sales_train_CC_items_CC_item_categories.merge(shops, 
                                                                                                 on="shop_id")

finalDF = sales_train_CC_items_CC_item_categories_CC_shops.copy()

# =============================================================================
# Extracting features
# =============================================================================
dateSplit = finalDF.date.str.split(".", expand=True)

dateSplit.columns = ["day", "month", "year"]

finalDF = pd.concat([finalDF, dateSplit], axis = 1)

# =============================================================================
# Dropping data
# =============================================================================
finalDF = finalDF[~(finalDF.item_cnt_day < 0)]
finalDF = finalDF[~(finalDF.item_price == -1)]
finalDF.drop(["item_name", 'shop_name', 'item_category_name'], axis = 1, inplace =True)

finalDF.date = pd.to_datetime(finalDF.date)
finalDF.item_cnt_day = finalDF.item_cnt_day.astype('int64')
finalDF.day = finalDF.day.astype('int64')
finalDF.month = finalDF.month.astype('int64')
finalDF.year = finalDF.year.astype('int64')
min_max_scaler = preprocessing.MinMaxScaler()
itemPrice = pd.DataFrame(min_max_scaler.fit_transform(finalDF.iloc[:, 4:6]),
                         columns=finalDF.iloc[:, 4:6].columns)
finalDF.item_price = itemPrice.item_price
a = finalDF.corr()
sns_plot = sns.pairplot(finalDF, hue='item_id')
sns_plot.savefig("item_id.png")
sns_plot = sns.pairplot(finalDF, hue='shop_id')
sns_plot.savefig("shop_id.png")
sns_plot = sns.pairplot(finalDF, hue='item_category_id')
sns_plot.savefig("item_category_id.png")
