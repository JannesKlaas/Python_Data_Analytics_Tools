#Jannes Klaas
#Python for data analytics
#Sample code for class 2

#My class 'Python for data analytics' teaches how to use Python for practical data analytics
#The class is mostly oriented to business students, but also to some other social sciences
#The code is to be run in an iPython notebook
#Some of the functions here are written in a less concise manner than they could be to showcase basic concepts
#All code is provided on an 'As is' basis, aka I take no responsibility if you screw up your homework with this



#As usual, import pandas and numpy first
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import scipy.stats as scs


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

#lets reduce the data a little bit so we only have to worry about the interesting variables
interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement', 'Emotional',
       'Calculative', 'Trust', 'Image']
       
reduced = data[interesting_collums]

###############################################################################
# Additional Univariate analysis stuff I will skip for now
# Look up the code, it us useful but I do not want to explain all of this now
from scipy.stats import zscore

#Compute all zscores
zscores = reduced.dropna().apply(zscore)

#Plot a single z score distribution
sns.distplot(zscores['Emotional'])

#Plot all histograms
import matplotlib as plt
for col_id in reduced.columns:
    #Run the seaborn plot, returns matplotlib axis
    plot = sns.distplot(reduced[col_id].dropna())
    #produce datapath to save item at
    #YOU WILL NEED TO CHANGE THIS ONE
    dest_string = '/Users/jannes/Google Drive/ABM/data/hist/'+col_id
    #Extract the actual figure from the axis
    fig = plot.get_figure()
    #Save it
    fig.savefig(dest_string)
    #Clear up matplotlib
    plt.pyplot.gcf().clear()
    
#Plot and save all distributions of z scores
for col_id in zscores.columns:
    #Ru the seaborn plot
    plot = sns.distplot(zscores[col_id].dropna())
    #Create file path
    #YOU WILL NEED TO CHANGE THIS ONE
    dest_string = '/Users/jannes/Google Drive/ABM/data/zscores/'+col_id
    #Extract and save figure, as above
    fig = plot.get_figure()
    fig.savefig(dest_string)
    #clear up matplotlib
    plt.pyplot.gcf().clear()

#Finally, boxplots
#Adjust font size so that we have nice readable graph
sns.set(font_scale = 0.8)
#and plot it
sns.boxplot(reduced)


####################################################################
####################################################################

#Univariate analysis
#Pandas plot docu: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
#plotting histograms is easy, just specify the series you would like
#and tell pandas to plot a histogram
reduced['Emotional'].plot(kind='hist')

#or simply 
reduced['Emotional'].hist()

#boxplots are also useful
reduced['Emotional'].plot(kind='box')


#We can test for normality easy
stat, pval = scs.mstats.normaltest(reduced['Emotional'])

#of course you can also just run these commands for the entire dataset
#but it can get messy
reduced.plot(kind='box')

#Lets compute a correlation matrix
corrmat = reduced.corr()
#you will see that the corrmat is just another pandas dataframe
#We can save it for later use
corrmat.to_excel('/Users/jannes/Google Drive/ABM/data/all_correlations.xls')

#To visualize the correlation matrix, we will use seaborn
#Seaborn is a very useful visualization module
#In class, we will not cover it in depth, read up on it here:
# http://seaborn.pydata.org/

#you install new modules with pip, 
#pip is run directly on the system command line, and not in python
#the ! tells rodeo to run the following command in the command line
!pip install seaborn

import seaborn as sns

#Now we can plot a heatmap of the correlations
#we pass as arguments the data and tell seaborn that
#the data is a square of relations, not just collums
sns.heatmap(corrmat, square=True)


import scipy.stats as scs

#Helper function which defines categories from the numerical values
#Every integer step becomes one category
#input: pandas series (aka one collum)
#output: a range object containing all categories
def categories(series):
    return range(int(series.min()), int(series.max()) + 1)
    
#Chi square function which performs an independence test between two
#collums of a data frame
def chi_square_of_df_cols(df, col1_name, col2_name):
    #extract the actual collums from the data frame
    df_col1, df_col2 = df[col1_name], df[col2_name]
    
    #the result matrix is a nested array of observed frequencies
    #visualize it as every list being a collum in the observed frequencies
    #and the whole nested list being a list of all collums
    #we just set this up to be empty here
    result = []
    #In the outer loop we go over every category for the first variable
    for cat1 in categories(df_col1):
        #we set up the inner list (the collums)
        cat1_value_list = []
        #now we loop over every category in the second variable
        for cat2 in categories(df_col2):
            #we count the number of occurences of cat1 and cat2 in the data
            num_measurements = len(df[ (df_col1 == cat1) & (df_col2 == cat2) ])
            #and append that value to the list
            cat1_value_list.append(num_measurements)
        #then we append the inner list (collum) to the overall nested list
        result.append(cat1_value_list)
    #now we let scipy stats run the chi square test
    chi2, p_value, df, expected = scs.chi2_contingency(result)
    #and return chi2, p value and degrees of freedom
    return chi2, p_value, df
    

chi_square_of_df_cols(data, 'loyalty', 'educ')

#Lets perform the chi2 test for every collum against every other
#we are not interested in all collums, just those 10
interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement', 'Emotional',
       'Calculative', 'Trust', 'Image']
n = len(interesting_collums)


#we save the resulting p values in a numpy matrix
import numpy as np
pval_mat = np.empty((n,n))

#Go to the collums
for i,name1 in enumerate(interesting_collums):
    #and for every collum go through all other collums
    for j, name2 in enumerate(interesting_collums):
        #perform the chi2 of those two
        chi2, p_value, df = chi_square_of_df_cols(data, name1, name2)
        #and just save the p value (we could use the other ones but nah)
        pval_mat[i,j] = p_value

#Save all the p vals as a text file which you can load e.g. in excel
pval_df = pd.DataFrame(pval_mat, columns=interesting_collums, index=interesting_collums)
pval_df.to_excel('/Users/jannes/Google Drive/ABM/data/all_chi2_pvals.xls')
np.savetxt('/Users/jannes/Google Drive/ABM/data/chi2_pvals.csv',pval_mat)

sns.heatmap(pval_df, square = True)



def t_test(subset1, subset2, collums_to_test):
    #Function which computes the t test for equal means for all collums specified
    #Input: subset1, subset2: DataFrames, split by some ordinal variable
    #Output: result_dataframe: a DataFrame containing p value, t statistic and the result of the f test as collums
    #... and the different variables which where tested as rows
    
    #First, we set up some dictionaries to temorarily hold the values before we move them to the dataframe
    f_test_dict = {}
    t_stat_dict = {}
    p_val_dict = {}
    #For every collum we are supposed to test...
    for collum in collums_to_test:
        #Check whether the collum is present in both subsets (if not just skip it)
        if not collum in subset1 or not collum in subset2:
            continue
        #Measure veriance in first subset
        df1_var = subset1[collum].var()
        #Measure sample size n in first subset
        df1_n = subset1[collum].count()
        #Same for the second subset
        df2_var = subset2[collum].var()
        df2_n = subset2[collum].count()
        #Compute the F statistic
        F = df1_var/df2_var
        #Compute the critical F value
        Fcritical = scs.f.ppf(0.95,df1_n-1,df2_n-1)
        #If F<Fcritical we have not enough evidence to reject H0: Equal variance
        equal_variance = (F<Fcritical)
        #We now compute the t test
        #dropna() just removes all empty cells, which makes the test run smooth
        #If a a single cell would be empty, the t test would just return Na 
        #Note how we use the result of the F test to tell the t test whether variance can be assumed equal 
        t_stat, p_value  = scs.ttest_ind(subset1.dropna()[collum],subset2.dropna()[collum],equal_var=equal_variance)
        #We then save the reuslts in our temporary dictionaries
        f_test_dict[collum]=equal_variance
        t_stat_dict[collum]=t_stat
        p_val_dict[collum]=p_value
    #After we have done all the t tests, we have to transform those dictionaries to a nice dataframe
    #First we turn the dictionary into a dataframe, this gives us all indices of the dictionary as rows
    pvdf = pd.DataFrame.from_dict(p_val_dict, orient='index')
    #since it does not give a collum name, we have to add it here
    pvdf.columns = ['p value']
    tstatdf = pd.DataFrame.from_dict(t_stat_dict, orient='index')
    tstatdf.columns = ['t stat']
    ftdf = pd.DataFrame.from_dict(f_test_dict, orient='index')
    ftdf.columns = ['Equal Variance']
    #Now we concentate all three data frames
    result_dataframe = pd.concat([pvdf,tstatdf,ftdf],axis=1)
    #and return the whole thing
    return result_dataframe

#usage
#Split data into subsets by gender
males = data[data['gender']==1]
#remove the gender collum form the subsets (since it is useless in the t test)
del males['gender']
females = data[data['gender']==2]
del females['gender']



#run function
result_dataframe = t_test(males,females,colls_to_test)

#save dataframe as excel file
result_dataframe.to_excel('/Users/jannes/Google Drive/ABM/data/t_test_by_gender.xls')

   
def seperated_dataframes(df, treatment):
    #Seperates dataframe into multiple by treatment
    #E.g. if treatment is 'gender' with possible values 1 (male) or 2 (female) 
    #the function returns a list of two frames (one with all males the other with all females)
    #Input: df: the dataframe to be split up
    #treatment: the name of the treament collums as a string
    #Output: list of all seperated dataframes
    
    #Obtain the collum that contains the treatment
    #TODO: add check whether treatment present in dataframe
    treat_col = data[treatment]
    #Init empty list to hold seperated dataframes
    dframes_sep = []
    #Go through all categories of the treatment
    for cat in categories(treat_col):
        #Select all rows that match the category for the treatmet into a new dataframe
        df = data[treat_col == cat]
        #append the selected dataframe
        dframes_sep.append(df)
    return dframes_sep

#Just some test gibberish
pv_dict = {}
for collum in colls_to_test:
    seperated_colls = []
    for df in dframes_sep:
        seperated_colls.append(df.dropna()[collum])
    fstat, pval = scs.f_oneway(*seperated_colls)
    pv_dict[collum]=pval

def one_way_anova(df, colls_to_test, treatment):
    #Function that performs a one way anover on all specified values for the specified treatment
    #Input: df: dataframe the operation is to be run on
    #colls_to_test: list of strings of collum names for which we are going to perform the anova
    #treament: string with treatment collum name
    #output: dataframe containing pvalue and fstat for all collums
    #The resulting dataframe has the collum names as rows and fstat and pval as collums
    
    #First, split the dataset by the treatment
    dframes_sep = seperated_dataframes(df, treatment)
    #Init an empty dictionary which will contain the pvalues
    pv_dict = {}
    #Init empty dictionary which will contain f statistics
    fstat_dict = {}
    #For every collum we want to run our anova on
    for collum in colls_to_test:
        #init empty array which is going to hold the colllum of interest from the seperated dataframes
        seperated_colls = []
        #for each of our previously seperated dataframes
        for df in dframes_sep:
            #obtain the collum of interest less all empty cells and add it to the list of all collums of interest
            seperated_colls.append(df.dropna()[collum])
        #run the one way anova for the sperated collums
        #usually the function f_oneway takes the different collums comma seperated
        #Like f_oneway(coll1, coll2, coll3)
        #The star tells it to treat the list we are passing as such comma seprated variables
        fstat, pval = scs.f_oneway(*seperated_colls)
        #save p value and fstat for this collum to the dict
        pv_dict[collum]=pval
        fstat_dict[collum] = fstat
    #Now we are going to turn those dictionaries into a dataframe
    #We want the collums in the dataframe to have usefull names
    #the convention we use is treatment_fstat or treatment_pval
    #that way we know what we have in there when we look at the dataframe later
    #first we generate the name of the fstat collum by connecting the treatment name with the string'_fstat'
    fstat_coll_name = treatment+'_fstat'
    #then we convert the fstat dict into a dataframe
    fstatdf = pd.DataFrame.from_dict(fstat_dict, orient='index')
    #and then we change the collum name in that dataframe
    fstatdf.columns = [fstat_coll_name]
    #same goes for pvals
    pval_coll_name = treatment+'_pval'
    pvdf = pd.DataFrame.from_dict(pv_dict, orient='index')
    pvdf.columns = [pval_coll_name]
    #then we connect the pval dataframe and the fstat dataframe
    result_dataframe = pd.concat([pvdf,fstatdf],axis=1)
    #and return the result
    return result_dataframe

interesting_collums = ['loyalty', 'satisfaction','educ', 'gender', 'age', 'Involvement', 'Emotional',
       'Calculative', 'Trust', 'Image']
res = one_way_anova(data, interesting_collums, 'loyalty')


all_data_frames = []
for collum in interesting_collums:
    res = one_way_anova(data, interesting_collums, collum)
    all_data_frames.append(res)

final = pd.concat(all_data_frames, axis=1)
final.to_excel('/Users/jannes/Google Drive/ABM/data/all_oneway_anova.xls')



########################################################
# Two way anova
# TODO: Extend for unbalanced design
# I could not find a good module implementation so this is an implementation from scratch
# Via: http://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
# Still not perfect so handle with care


data = data.dropna()
#treat1: supp treat2: dose
def two_way_anova(data, measurement, treat1, treat2):
    #Counting all measurements
    N = len(data[measurement])
    a = len(data[treat1].unique())
    b = len(data[treat2].unique())
    r = N/(a*b)
    print(a,b,r)
    #df_a = a - 1 (a # categories a )
    df_a = a - 1
    #df_b = b-1 (b # categories b)
    df_b = b - 1
    # df_axb = (a-1)*(b-1) = df_a*df_b
    df_axb = df_a*df_b
    #df_w (for MSE) = N-a*b (a,b #categories a, b)
    df_w = N - (len(data[treat1].unique())*len(data[treat2].unique()))
    #grand mean = mean for all measurements
    grand_mean = data[measurement].mean()
    
    #The mean of measurement where treat1 is l less the grand mean for all l
    ssq_a = sum([(data[data[treat1] ==l][measurement].mean()-grand_mean)**2 for l in data[treat1]])
    
    ssq_b = sum([(data[data[treat2] ==l][measurement].mean()-grand_mean)**2 for l in data[treat2]])
    print(len(data[measurement]), grand_mean)
    ssq_t = sum((data[measurement] - grand_mean)**2)
    ssq_w = 0
    for cat in data[treat1].unique():
        vc = data[data[treat1] == cat]
        vc_dose_means = [vc[vc[treat2] == d][measurement].mean() for d in vc[treat2]]
        ssq_w += sum((vc[measurement] - vc_dose_means)**2)
       
    ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w
    ms_a = ssq_a/df_a
    ms_b = ssq_b/df_b
    ms_axb = ssq_axb/df_axb
    ms_w = ssq_w/df_w
    f_a = ms_a/ms_w
    f_b = ms_b/ms_w
    f_axb = ms_axb/ms_w
    p_a = scs.f.sf(f_a, df_a, df_w)
    p_b = scs.f.sf(f_b, df_b, df_w)
    p_axb = scs.f.sf(f_axb, df_axb, df_w)
    results = {'sum_sq':[ssq_a, ssq_b, ssq_axb, ssq_w],
           'df':[df_a, df_b, df_axb, df_w],
           'F':[f_a, f_b, f_axb, 'NaN'],
            'PR(>F)':[p_a, p_b, p_axb, 'NaN']}
    columns=['sum_sq', 'df', 'F', 'PR(>F)']

    aov_table1 = pd.DataFrame(results, columns=columns,
                              index=[treat1, treat2, 
                              treat1+':'+treat2, 'Residual'])
    return aov_table1

two_way_anova(data,'loyalty','Trust','Emotional')    
    


