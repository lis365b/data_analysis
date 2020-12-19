# Cheat Sheet: Data Preparation With Python

## Summary

Quick reference - preparing data for analysis using Python 3 and Jupyter Notebooks.

Version 1.0

Feedback welcome - elisabeth.reitmayr@gmail.com

**To be added in the future:**

   * Window functions
   * OneHotEncoder

## Sources

  * [Pandas documentation](http://pandas.pydata.org/pandas-docs/stable/)
  * [Scipy Documentation](https://docs.scipy.org/doc/numpy/reference/)
  * [Notebook on data pre-processing (April Chen)](https://github.com/aprilypchen/depy2016/blob/master/DePy_Talk.ipynb)
  * [Awesome Pandas blog](https://chrisalbon.com/) 

# Table of contents
   * [Import libraries](#H1)
   * [Import data](#H2)
      * [Read from csv](#H2a)
      * [Connect to a database](#H2b)
      * [Use a date picker to pull data from database](#H2c)
  * [Styling](#H3)
     * [Hide warnings](#H3a)
     * [Toggle code](#H3b)
     * [Table of contents](#H3c)
     * [Change the maximum columns/column width to be displayed](#H3d)
  * [Metadata and data types](#H4)
     * [Metadata](#H4a)
     * [Modify data types](#H4b)
  * [Duplicate detection](#H5)
     * [Find duplicates](#H5a)
     * [Remove duplicates](#H5b)
  * [Dataframe manipulation](#H6)
     * [Columns and rows](#H6a)
     * [Aggregation, filtering and sorting](#H6b)
     * [Combine dataframes](#H6c)
  * [Outlier detection](#H7)
     * [Tukey’s test for extreme values](#H7a)
     * [Kernel density estimation](#H7b)
  * [Variable generation and manipulation](#H8)
     * [Generate new variables](#H8a)
     * [Bucket variables](#H8b)
     * [Encode categorical variables](#H8c)
     * [Generate dummy variables](#H8d)
  * [Prepararation of data for modeling](#H9)
     * [Draw samples and split dataset](#H9a)
     * [Reshape data for modeling](#H9b)

<a name="H1"></a>
## Import libraries

```
import pandas as pd
import numpy as np
import psycopg2 as ps
from IPython.display import HTML
%matplotlib inline # show plots inline in Jupyter Notebook
```
<a name="H2"></a>
## Import data
<a name="H2a"></a>
### Read from csv

```
df = pd.read_csv('PATH', encoding = 'utf-8-sig') 
len(df) # N rows imported (add for future reference)
```

Do not forget to define encoding as this might cause issues. If you store the csv file in the same folder as the Notebook, it is sufficient to specify the file name.

If you want dates to be recognized as dates, add this argument:

```
pd.read_csv('fileName.csv', parse_dates=True)
```

<a name="H2b"></a>
### Connect to a database

```
# Establish database connection
con=ps.connect(dbname= 'DBNAME', host='HOST', 
port= 'PORT', user= 'USER', password= 'PASSWORD')
cur = con.cursor()

# Query raw data
cur.execute("""SELECT STATEMENT""")
data = cur.fetchall()
len(data) # Nrows imported, DATE (add for future reference)

# Close connection
cur.close()

# Parse the tables into a Pandas dataframe (specify column names)
data = pd.DataFrame(list(data), columns=('COL1', 'COL2', 'CO3'))
data.head()
```

Please note: You might want to consider security precautions when pulling your data directly from your database (SSH tunneling etc.).

<a name="H2c"></a>
### Use a date picker to pull data from database

Download [ipywidgets](https://github.com/jupyter-widgets/ipywidgets), copy the folder "ipywidgets" into the directory of your notebook and execute the following code snippet from you notebook.


```
from ipywidgets import widgets
from IPython.display import display
from IPython.display import clear_output

w1 = widgets.DatePicker()
display(w1)
date1 = "2017-02-10"
date2 = "2017-02-10"

def on_value1_change(change):
    global date1
    date1 = change['new'].strftime('%Y-%m-%d')
    ask_datavault()

w1.observe(on_value1_change, names='value')

w2 = widgets.DatePicker()
display(w2)
def on_value2_change(change):
    global date2
    date2 = change['new'].strftime('%Y-%m-%d')
    ask_datavault()

w2.observe(on_value2_change, names='value')

def ask_datavault():
    clear_output()
    # Establish database connection
    con=ps.connect(dbname= 'NAME', host='HOST', 
    port= 'PORT', user= 'USER', password= 'PASSWORD')
    cur = con.cursor()
    # Query raw data
    cur.execute("SELECT STATEMENT WHERE date BETWEEN '" + date1 + "' AND '" + date2 + "' GROUP BY date")
    data = cur.fetchall()
    # Close connection
    cur.close()
    print(data)
```

<a name="H3"></a>
## Styling
<a name="H3a"></a>
### Hide warnings

If you want to hide warning messages, `import warnings` and add `warnings.filterwarnings('ignore')` at the top of your notebook (be aware you will not be warned if you use deprecated methods etc. in case you do this).

<a name="H3b"></a>
### Toggle code
Download [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) and copy the folder "ipywidgts" into the directory from which you start your notebook. Then, execute the following code snippet from a notebook cell.


```
from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this Jupyter notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
```

<a name="H3c"></a>
### Table of contents

For ipynb files, you can use the table of contents options (button in your toolbar, cannot be exported to HTML etc. per default).

You can use an extension to display the auto-generated table of contents "inside" your notebook and to make it exportable - read more [here](http://www.sas-programming.com/2016/08/add-table-of-contents-to-your-jupyter.html).

Follow the instructions to add your table of contents in a cell. If the table of contents does not show up, click on the refresh symbol in the "Contents" side menu in your notebook.

<a name="H3d"></a>
### Change the maximum columns/column width to be displayed

Column width:

```
pd.set_option('max_colwidth',800)
```

Number of columns:

```
pd.set_option("display.max_columns",200)
```
<a name="H4"></a>
## Metadata and data types
<a name="H4a"></a>
### Metadata

**Runtime info for single cells:** `%%time` magic gets you the time spent on cell execution.

**Info on data types:** 

```
df.info()
```  

**Summary of missing values:** 

```
df.isnull().sum()
``` 

**Replace missings by 0 for numeric variables (if appropriate):**

```
for i in range(0, 3): 
		df.iloc[:,i].fillna(value=0, inplace=True)
```

Specify the range of columns - here: columns 1-4.

<a name="H4b"></a>
### Modify data types

**Cast to other data types:**

```
for i in range(1, 2):   
	    df.iloc[:, [i]] = df.iloc[:, [i]].apply(np.int64) 
```
Specify the range of columns (here: columns 2-3) and the data type (can also be String, etc.).

**Convert a microseconds integer timestamp to datetime:**

```
df.my_ts = pd.to_datetime(df.my_ts)
```

<a name="H5"></a>
## Duplicate detection
<a name="H5a"></a>
### Find duplicates

**Count unique values of one column:**

```
df.groupby('my_category').my_user_id.nunique()
```

**Show duplicate values for one column:**

```
pd.concat(i for _, i in df.groupby('my_user_id') if len(i) > 1)
```

In case you want to show duplicates for a combination of columns, replace 'my_user_id' by the list of columns to be considered (e.g. ['my_user_id', 'my_country']).

<a name="H5b"></a>
### Remove duplicates

**Remove duplicates:**

```
df = df.drop_duplicates(subset='my_user_id', keep=False)
```  

Change 'keep' parameter to 'first'/'last' if applicable.

<a name="H6"></a>
## Dataframe manipulation
<a name="H6a"></a>
### Columns and rows

**Drop one or more  column(s):**

```
df = df.drop(['col1', 'col2'], axis=1)
```

The axis argument specifies that you want to drop columns, not rows.

**Delete a column:** 
```
del df['COL1']
```

**Drop one or more rows:** 

```
df = df.drop(['Index1', 'Index2'])
```

Specify the index/row numbers to drop.

**Remove all rows that do not fulfill a condition:**

```
df = df[df.COL1 < x]
```

x is a numeric threshold in this example, can be modified to any other condition.

**Replace values:** 
```
df[COL1].replace({"VALUE1": 0, "VALUE2": 1})
```

Specify the values to be replaced in "".

**Rename one column:**

```
df = df.rename(columns = {'OLDNAME': 'NEWNAME'})
```


**Rename all columns:** 

```
df.columns = ['COL1', 'COL2', 'COL3']
```

**Change the order of columns:**

```
# Show the list of columns to then copy and rearrange
cols = list(df.columns.values)
# Rearrange
df = df[['COL2', 'COL3', 'COL1']]
```

**Transpose a dataframe:**

```
data = data.transpose(as_index = False)
```

Chose whether to transpose the columns as index or not.

**Pivot a dataframe:**

```
df2 = df.pivot_table('CATEGORY_COUNT', 'INDEX_VARIABLE', 'CATEGORY')
```

The dataframe will be reshaped such that the values of a categorical variable become columns. 

<a name="H6b"></a>
### Aggregation, filtering and sorting

**Aggregate by grouping (equivalent to SQL "GROUP BY"):**

```
df.groupby(by = ['COL1', 'COL2'], as_index = False, sort = False).COL3.sum()
```

Specify aggregation function - can also be values_count() etc.

In this example, COL3 will be aggregated by the sum over COL1 and COL2.

**Filter a dataframe:**

One condition:  
```
df[df.COL1 > 1]
```


Multiple conditions:
```
df[(df['colCOL11'] >= 1) & (df['COL2'] <=1 )]
```

**Sort dataframe:**

```
df.sort_values(by = ['COL1', 'COL2'], ascending = (0, 1))
```

In this example, COL1 will be sorted descendingly, COL2 will be sorted ascendingly.

**Add a column containing the sum over different columns:**

```
helplist = ['COL1', 'COL2', 'COL3']
df['total'] = df[helplist].sum(axis = 1)
``` 

In this example, a new column called "total" that contains the sum over COL1-3 for each row will be added to the dataframe.

<a name="H6c"></a>
### Combine dataframes

**Join dataframes (equivalent to SQL JOIN operations):**

```
df = pd.merge(df1, df2, how = 'left', left_on = ['my_user_id'], right_index = True)
```
Change join type and "on" argument to the applicable variables or to the index.

**Add rows of another dataframe to your dataframe:**

```
df = pd.concat([df1, df2], axis = 0)
```

<a name="H7"></a>
## Outlier detection

[Source](https://github.com/aprilypchen/depy2016/blob/master/DePy_Talk.ipynb)

<a name="H7a"></a>
### Tukey's test for extreme values

```
# Define function using 1.5x interquartile range deviations from quartile 1/3 as floor/ceiling
def find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1 
    floor = q1 - 1.5 * iqr
    ceiling = q3 + 1.5 * iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values

# Print outliers for each numeric variable
for x in range(1, 7): # Modify to select numeric columns
    tukey_indices, tukey_values = find_outliers_tukey(data.ix[:, x])
    print(list(data[[x]]), np.sort(tukey_values))
```
<a name="H7b"></a>
### Kernel density estimation 

Non-parametric, captures also bimodal distributions.

```
from statsmodels.nonparametric.kde import KDEUnivariate

# Define outlier function
def find_outliers_kde(x):
    x_scaled = scale(list(map(float, x)))
    kde = KDEUnivariate(x_scaled)
    kde.fit(bw = "scott", fft = True)
    pred = kde.evaluate(x_scaled)
    
    n = sum(pred < 0.05)
    outlier_ind = np.asarray(pred).argsort()[:n]
    outlier_value = np.asarray(x)[outlier_ind]

    return outlier_ind, outlier_value

# Print outlier values
for x in range(1, 7): # Modify to select numeric columns
    kde_indices, kde_values = find_outliers_kde(data.ix[:, x])
    print(list(data[[x]]), np.sort(kde_values))
```

<a name="H8"></a>
## Variable generation and manipulation
<a name="H8a"></a>
### Generate new variables

**Add a column to a dataframe:**

```
df['NEWCOL'] = 0
```

Specify the values to be inserted - can be done through aritmethic operations on other columns, e.g.:

```
df['NEWCOL'] = df.COL1/df.COL2
```

If you want to apply a function to another variable of the dataframe, you will have to map it to this variable:

```
def mins_to_secs(x):
	x = df.time_spent_minutes/60

df['time_spent_seconds'] = df.time_spent_minutes.map(mins_to_secs)
```

<a name="H8b"></a>
### Bucket variables

**Bucket several categorical values into one value:**

Example: You have a categorical variable "countries" that has 180 unique values. You want to see the 3 most important countries only and bucket all other countries into a value "other".

```
# Print unique values of each categorical variable in data
for i in df.columns:
    if df[i].dtypes=='object': 
        unique_cat=len(df[i].unique())
        print("Feature '{i}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))

# Print counts of each value of categorical variable
print(df['COUNTRY'].value_counts())

# Bucket low frequency categories as "Other"
def repl(x):
    if x == 'US': return 'US'
    elif x == 'BR': return 'BR'
    elif x == 'ES': return 'ES'
    else: return 'Other'
    
df['COUNTRY'] = df['COUNTRY'].apply(repl)
print(df['COUNTRY'].value_counts().sort_values(ascending=False))
```

**Bucket numeric variables into categories**

**Equal-sized bins:**

```
df['var'] = 0
var = pd.qcut(x=COL1, q=3, labels=["good", "medium", "bad"])
print(df['var'].value_counts().sort_values(ascending=False))
```
Adjust the number of bins (q) and the labels.

**Equal-intervaled bins:**

```
bins = [0, 20, 40, 60, 80, 100] 
df.var = pd.cut(x=COL1, bins, labels=['Very low', 'Low', 'Medium', 'High', 'Very high']) 
print(data['var'].value_counts().sort_values(ascending = False))
```
Adjust the bins by specifying the thresholds for the bins 
(must be one more than the number of categories/labels). Per default, bins include rightmost edge, set argument "right" =False if rightmost edges shall not be included.

<a name="H8c"></a>
### Encode categorical variables

**Encode a boolean variable by casting to integer:**

```
df['BOOL'] = (df.COL1=="ABC").astype(int)
```
In this example, COL1 contains string values. BOOL will be 1 if COL1 contains "ABC".

**Encode manually by mapping a dictionnary:**

```
dic = {'Yes': 1, 'No': 2}
df['VAR'] = df['VAR'].map(dic)
```

**Encode automatically:**

to be added (OneHotEncoder)

<a name="H8d"></a>
### Generate dummy variables

**For a non-binary categorical variable:**

```
dummy = pd.get_dummies(dta1['country'], prefix='ct').astype(int)
dummy.head()
```

**For all categorical variables of a dataframe:**

```
# Create a list of features to dummy
dummy_vars = ['COL1', 'COL2', 'COL3']

# Create dummies for all categorical variables
def dummy_data(df, dummy_vars):
    for x in dummy_vars:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

df = dummy_df(df, dummy_data)
print(df.head())

# Dummies
dummy_ct = pd.get_dummies(df1['country'], prefix='ct').astype(int)
dummy_pf = pd.get_dummies(df1['platform'], prefix='pf').astype(int)

# Define columns to keep
colstokeep = ['ABC', 'DEF']

# Join dummies to columns to be kept from original dataframe using an identifier that is not in the list of columns to keep
df1 = df1[colstokeep].join(dummy_ct.ix[:, 'ct_BR':]).join(dummy_pf.ix[:, 'pf_amazon':])
df1.head()
```

<a name="H9"></a>
## Preparation of data for modeling
<a name="H9a"></a>
### Draw samples and split dataset

**Draw a random sample from a dataset:**

```
data2 = data1.sample(1000)
```

Axis is per default 0 (rows).

**Split test and training data:**

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
```

Adjust size of test dataset.

<a name="H9b"></a>
### Reshape data for modeling

**Define dependent and independent variables:**

Reshape dataframe to array (input for Scikit-Learn or Scipy models): 

```
array = df.values
```

Define variables:

```
X = array[:,1:5]
Y = array[:,0]
```

Specify columns according to your dataframe, here: X: columns 2-6.

**Flatten dataframe into a 1-dimensional array:** 

```
Y = np.ravel(Y)
```
