#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# Machine learning algorithm using Elastic-Net regression.
# 
# ## Import Libraries

# In[1]:


import pandas as pd # library for data manipulation (dataframes)
import numpy as np # library for computations
import matplotlib.pyplot as plt # library for visualizatio
from sklearn.linear_model import ElasticNet  # library for machine learning methods
from sklearn.linear_model import ElasticNetCV# library for machine learning methods
import xlsxwriter as xlsw # library for exporting excel files


# ## Definition for Importing Data

# In[2]:


def import_data(file_location, sheet):
    """This function returns data read from an xlsx file from a specific table.
    The sheet inside the file must include a '#' as the origin of the table.
    """
    #read data table from file
    values = pd.read_excel(file_location, sheet_name=sheet)
    #set origin of table at '#'
    values.set_index('#', inplace=True)
    return values


# ## Import Trainning Data

# Assign address for trainning data table to variable ('x' values).
# This set represents quantitative data for a group of units.
# In this example the units are buildings and the data for each
# building are NORMALIZED measurements such as areas, connectivity
# between spaces, visibility inside spaces, etc.

# In[3]:


x_values = import_data('quant_data_example.xlsx', 'trainning_set')


# Print data table.

# In[4]:


x_values


# ## Import Labels

# Assign address for data table from labels to variable ('y' values).
# This set represents the labels by which the quant data
# will be trainned. In this case these are scores extracted
# from a survey, using a phenomenological approach, about
# the different buildings from 1 to 5.
# The scores were separated into different fields such as
# orientation, circulation, hierarchy of spaces, etc.

# In[5]:


y_values = import_data('labels_example.xlsx', 'trainning_labels')


# Print data table.

# In[6]:


y_values


# ## Import Testing Data

# Read data table from excel file.

# In[7]:


testing_data = import_data('quant_data_example.xlsx', 'testing_set')


# Print data table.

# In[8]:


testing_data


# # ML Models (Elastic Net)

# ## Elastic Net Definition

# In[9]:


def elastic_net(x, y, cross_validation):
    """This function returns a trainned model from a data set ('x') based on a label ('y').
    The variable 'round_integer' defines the precision for coefficients for each variable 'x' for every 'y'.
    The variable 'cross_validation' defines the number of group samples k for cross validation.
    """
    trainned_model_cross = ElasticNetCV(cv=cross_validation, random_state=0).fit(x, y)
    trainned_model = ElasticNet(alpha = trainned_model_cross.alpha_).fit(x, y)
    return trainned_model


# ## Prediction

# Taking the label "Hierarchy" from the example labels, lets train the model and predict the score of the test building for "Hierarchy".

# In[10]:


trainned_model = elastic_net(x_values, y_values.Hierarchy, 6)


# Plot the coefficients of the trainned model

# In[11]:


np.round(trainned_model.coef_, 4)


# Predict the score test building

# In[12]:


trainned_model.predict(testing_data)


# # Conclusion

# After trainning the model for the specific label "Hierarchy", the score for that label for the test building is 3.13 out of 5.00

# In[ ]:




