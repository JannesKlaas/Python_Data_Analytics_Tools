#As usual, import modules first
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import scipy.stats as scs

import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm

#also as usual, lets load the data
#I converted the given SPSS file into a CSV file so we can work with it 
#The file should be contained in the same folder
data = pd.read_csv('/Users/jannes/Google Drive/ABM/data/case.csv')

#Lets quickly check whether the import went well
data.dtypes

#The educ collum seems to be in trouble, there seem to be some values
#missing, so pandas set the data type to object
#it needs to be a number (either integer or float)
#so lets convert it 

#data['educ'] is how we adress the education collum in the dataframe
#pd.to_numeric is the pandas conversion function
#errors='coerce' means that we force the conversation.
#Values that can not be converted (usually because they are empty)
#are set to NaN, Not a Number, we dont ususally have to worry about them
data['educ'] = pd.to_numeric(data['educ'],errors='coerce')

#The education values have been converted to floats
data.dtypes


#All educ values that are not numbers (aka empty) get saved as Nan, not a number
#We can count the number of missing values
data.isnull().sum()

#lets just drop them
data = data.dropna()

#and limit us again to a reduced dataset
interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement', 'Emotional',
       'Calculative', 'Trust', 'Image']

reduced = data[interesting_collums]

#Simple linear regression
res = sm.ols(formula="loyalty ~ Trust", data=reduced).fit()

#we can view a summary
res.summary()

#or we can just check the parameters (for further computation)
res.params

#or maybe just a single interesting value...
res.rsquared

#or the p values, there is so much we could use for computation...
res.pvalues

#multivariate regression
res = sm.ols(formula="loyalty ~ Trust + Image", data=reduced).fit()
res.summary()

#we can also directly apply operators
#Lets run a linear regression against squared values
res = sm.ols(formula="loyalty ~ Trust**2", data=reduced).fit()
res.summary()

#We write python code for rapid testing, so how can we automate this?

#statsmodels can do more useful stuff, like anova
model = sm.ols(formula='loyalty ~ C(Calculative)', data=reduced).fit()
table = anova_lm(model)
table



#or two way anova!
model = sm.ols(formula='loyalty ~ C(Trust) * C(Image)', data=reduced).fit()
table = anova_lm(model)
table

#Generating formula strings

dependent = 'loyalty'
independent = 'loyalty'
f = dependent + ' ~ ' + independent

res = sm.ols(formula=f, data=reduced).fit()
res.summary()

#To win friends in IBA, I like to run the stats for every collum against every other collum 
#and then post them on facebook
#note: this is slightly silly and you usually would not do it
folder = '/Users/jannes/Google Drive/ABM/class3/lin_reg_reports/'
anova_folder = '/Users/jannes/Google Drive/ABM/class3/one_way_anova_reps/'
for dependent in reduced.columns:
    for independent in reduced.columns:
        #Generating formula string
        f = dependent + ' ~ ' + independent
        #we want to give the files a slightly nicer name
        file_ending = independent + ' by ' + dependent + '.txt'
        file_path = folder+file_ending
        res = sm.ols(formula=f, data=reduced).fit()
        anova_formula = dependent + ' ~ ' + 'C(' + independent+ ')'
        anova_result = sm.ols(formula=anova_formula, data=reduced).fit()
        anova_table = anova_lm(anova_result)
        anova_table.to_excel(anova_folder + dependent + ' by ' + independent + '.xls')
        text = res.summary().as_text()
        file = open(file_path, 'w')
        file.write(text)
        file.close()



#Example: How to work with such a dataset?

#Step 1: Read the description!

#step 2: exploratory analysis

#univariate
#Adjust font size so that we have nice readable graph
sns.set(font_scale = 0.8)
#and plot it
sns.boxplot(reduced)


#Our scatterplots from the first time do not tell us much here, since it is all integer values
#We need the density of the scatter plot!
#Lets plot a kernel density estimator

#Since we run this in an iPython notebook, we have to use the inline command
#It is a little bit of black magic, we just have to go and execute the following lines all at once
#Warning: this might take a while
%matplotlib inline
#set seaborn to default settings
sns.set()
#Start an empty pair grid
g=sns.PairGrid(reduced)
#on the diagonals we are going to have some univariate kernel density estimation
g.map_diag(sns.kdeplot)
#On the diagonals we have bivariate kernel density estimation
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);

#It is often useful to give some relationships a closer look
sns.set()
g = sns.jointplot("educ", "loyalty", data=reduced, kind="reg", color="b")

#Violin plots are a good way to not learn about distributions
sns.set_style("whitegrid")
sns.violinplot(y='loyalty', data=reduced)

#also distributions that are affected by independent variables
sns.violinplot(x='age', y='loyalty', hue='gender', data=reduced, split=True)

#Interaction plots are a great way to learn about multivariate interactions
sns.interactplot(x1='Emotional', x2='Trust', y='loyalty', data=reduced)

#Step 3 
#Think! Come up with some hypothesis!

#Step 4 Test
#the usual social science tests

#Hypothesis: Emotional people are more loyal

#Simple linear regression
res = sm.ols(formula="loyalty ~ Emotional", data=reduced).fit()

res.summary()

#Answer: Yes they are but that is not enought to explain loyalty

#Step 4: Build model

#Maybe loyalty is just a function of emotionality, calculativeness, and satisfaction....
#Then a linear regression would do the trick!
#We just need to know the co efficioents, easy (also pretty boring)
res = sm.ols(formula="loyalty ~ Emotional + satisfaction + Calculative", data=reduced).fit()
res.summary()

#Step 5: test model
#In ML we usually have seperate test (also called cross validation) sets
#Here we are just going to use our training set 

#we can simply output all residuals
res.resid
