#!/usr/bin/env python
# coding: utf-8

# # Intern Name = Ankesh kumar Intern ID =IN9240927  Project = Performing EDA ON AMCAT Data

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv(r"C:\Users\ankes\OneDrive\Desktop\Data\data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[39]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


df.nunique()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


# Univariate Analysis
# PDFs, Histograms, and Boxplots for numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for column in numerical_columns:
    plt.figure(figsize=(12, 6))

    # Probability Density Function (PDF)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

    # Identify outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    print(f'Outliers in {column}:\n{outliers}')

    # Frequency distribution for numerical columns
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=20, kde=False)
    plt.title(f'Frequency Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# In[14]:


##observations
# Columns  :    Observations
#Salary   :    Distribution is right skewed and there is very less outlier data .
#10percentage : distribution is nearly normally distributed and there is large outlier .
#12graduation : distribution is nearly normal distribution  and have few outlier
#12percentage : iit has is normal distribution .
#collegeGPA   : it has  is normal distribution with lots of outlier .
#English      : it has  is normal distribution with lots of outlier .
#Logical      :it has  is normal distribution with few of outlier .
#Quant        : it has  is normal distribution with very little of outliers .
#ComputerProgramming:  it has  is normal distribution with few outliers.
#conscientiousness  :  it has  is normal distribution with lots of  outliers.
#agreeableness:  it has  is normal distribution with lots of outliers.
#extraversion :  it has  is normal distribution with lots of outliers.
#nueroticsm   :  it has  is normal distribution with few outliers.
#openess_to_experience   : it has  is left skewed distribution with lots of outliers.


# In[15]:


# Countplots for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    plt.figure(figsize=(15, 6))
    sns.countplot(x=df[column])
    plt.title(f'Countplot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


# In[16]:


# Countplot with aggregated categories
column='Specialization'
top_categories = df[column].value_counts().nlargest(10)  # Adjust the number of top categories as needed
plt.figure(figsize=(12, 8))
sns.barplot(x=top_categories.index, y=top_categories.values)
plt.title(f'Countplot of Top Categories in {column}')
plt.xlabel(column)
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[17]:


# Countplot with aggregated categories
column='CollegeState'
top_categories = df[column].value_counts().nlargest(10)  # Adjust the number of top categories as needed
plt.figure(figsize=(12, 8))
sns.barplot(x=top_categories.index, y=top_categories.values)
plt.title(f'Countplot of Top Categories in {column}')
plt.xlabel(column)
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility


# In[18]:


#Observation:
# In Gender column there is three times male compared to females.
#In Degree column mostly people took  B.Tech/B.E. as there degree.
#In specialization most of them took electronics and communication engineering followed by computer science and information technology and then computer engineering
#Most of peoples college state is Uttar Pradesh


# In[19]:


# Univariate Analysis - Salary Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Salary'], bins=20, kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.show()


# In[20]:


# Univariate Analysis - 10th and 12th Percentages
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['10percentage', '12percentage']])
plt.title('10th and 12th Percentages Distribution')
plt.show()


# In[21]:


# Univariate Analysis - Top 10 Specializations Distribution
top_10_specializations = df['Specialization'].value_counts().nlargest(10).index

plt.figure(figsize=(12, 6))
sns.countplot(x='Specialization', data=df[df['Specialization'].isin(top_10_specializations)])
plt.title('Top 10 Specializations Distribution')
plt.xlabel('Specialization')
plt.xticks(rotation=45)
plt.show()


# In[22]:


# Univariate Analysis - Personality Traits
traits = ['conscientiousness', 'agreeableness', 'extraversion', 'nueroticism', 'openess_to_experience']
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[traits])
plt.title('Personality Traits Distribution')
plt.show()


# In[23]:


# Univariate Analysis - Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()


# In[24]:


# Display a histogram and a boxplot for the Salary distribution
plt.figure(figsize=(12, 6))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(df['Salary'], bins=20, kde=True, color='skyblue')
plt.title('Salary Distribution - Histogram')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Salary'], color='salmon')
plt.title('Salary Distribution - Boxplot')

plt.tight_layout()
# Save the figure
plt.savefig('salary_distribution_plot.png')
plt.show()

# Observations
# Note any skewness or asymmetry
skewness = df['Salary'].skew()
print(f"Skewness: {skewness}")

# Identify and discuss outliers
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Salary'] < Q1 - 1.5 * IQR) | (df['Salary'] > Q3 + 1.5 * IQR)]
print(f"Number of outliers: {len(outliers)}")
print(outliers[['Salary']])


# In[25]:


# Scatter Plot for Salary vs. ComputerProgramming scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ComputerProgramming', y='Salary', data=df, color='red')
plt.title('Bivariate Analysis - Salary vs. ComputerProgramming Scores')
plt.xlabel('ComputerProgramming Scores')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[26]:


# Scatter Plot for Salary vs. Quant
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quant', y='Salary', data=df, color='red')
plt.title('Bivariate Analysis - Salary vs. Quant')
plt.xlabel('Quant')
plt.ylabel('Salary')
plt.show()


# In[27]:


# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Display the lists of numerical and categorical columns
print("Numerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)


# In[28]:


# Select the relevant numerical columns for analysis
numerical_columns = ['Salary', '10percentage', '12graduation', '12percentage', 'collegeGPA', 'English', 'Logical', 'Quant',
                     'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience', 'MechanicalEngg',
                     'ElectricalEngg', 'TelecomEngg', 'CivilEngg', 'conscientiousness', 'agreeableness', 'extraversion',
                     'nueroticism', 'openess_to_experience']

# Plot each numerical column against 'Salary'
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column, y='Salary', data=df)
    plt.title(f'Scatter Plot of {column} vs Salary')
    plt.xlabel(column)
    plt.ylabel('Salary')
    plt.show()


# In[29]:


##observation
#As attritbutes on x axis value increases salary is concentrated at some points


# In[30]:


# Select the top N designations
top_designations = df['Designation'].value_counts().nlargest(10).index
df_filtered = df[df['Designation'].isin(top_designations)]

# Set the style of seaborn
sns.set(style="whitegrid")

# Swarmplot
plt.figure(figsize=(14, 8))
sns.swarmplot(x='Salary', y='Designation', data=df_filtered)
plt.title('Swarmplot of Designation vs Salary (Top 10 Designations)')
plt.xlabel('Salary')
plt.ylabel('Designation')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# Boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(x='Salary', y='Designation', data=df_filtered)
plt.title('Boxplot of Designation vs Salary (Top 10 Designations)')
plt.xlabel('Salary')
plt.ylabel('Designation')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# Barplot
plt.figure(figsize=(14, 8))
sns.barplot(x='Salary', y='Designation', data=df_filtered)
plt.title('Barplot of Designation vs Salary (Top 10 Designations)')
plt.xlabel('Salary')
plt.ylabel('Designation')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[31]:


#observation
#senior software engineer has maximum salary
#technical support engineer has minimum salary


# In[32]:


# Set the style of seaborn
sns.set(style="whitegrid")

# Swarmplot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Gender', y='Salary', data=df)
plt.title('Swarmplot of Gender vs Salary')
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Salary', data=df)
plt.title('Boxplot of Gender vs Salary')
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()

# Barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='Salary', data=df)
plt.title('Barplot of Gender vs Salary')
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()


# In[33]:


##observation
#both the genders have nearly same salary


# In[34]:


# Set the style of seaborn
sns.set(style="whitegrid")

# Swarmplot
plt.figure(figsize=(12, 6))
sns.swarmplot(x='Degree', y='Salary', data=df)
plt.title('Swarmplot of Degree vs Salary')
plt.xlabel('Degree')
plt.ylabel('Salary')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Degree', y='Salary', data=df)
plt.title('Boxplot of Degree vs Salary')
plt.xlabel('Degree')
plt.ylabel('Salary')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# Barplot
plt.figure(figsize=(12, 6))
sns.barplot(x='Degree', y='Salary', data=df)
plt.title('Barplot of Degree vs Salary')
plt.xlabel('Degree')
plt.ylabel('Salary')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[35]:


##observation
#M.tech/M.E. students have highest salary


# In[36]:


# Set the style of seaborn
sns.set(style="whitegrid")

# Swarmplot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Gender', y='collegeGPA', data=df)
plt.title('Swarmplot of Gender vs collegeGPA')
plt.xlabel('Gender')
plt.ylabel('collegeGPA')
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='collegeGPA', data=df)
plt.title('Boxplot of Gender vs collegeGPA')
plt.xlabel('Gender')
plt.ylabel('collegeGPA')
plt.show()

# Barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='Gender', y='collegeGPA', data=df)
plt.title('Barplot of Gender vs collegeGPA')
plt.xlabel('Gender')
plt.ylabel('collegeGPA')
plt.show()


# In[37]:


##observation
#ColllegeGPA of females are slightly more than males


# In[38]:


# Scatter Plot for Salary vs. College GPA
plt.figure(figsize=(10, 6))
sns.scatterplot(x='collegeGPA', y='Salary', data=df, color='skyblue')
plt.title('Bivariate Analysis - Salary vs. College GPA')
plt.xlabel('College GPA')
plt.ylabel('Salary')
plt.show()


# In[40]:


df.columns


# In[41]:


# Select categorical columns for analysis
categorical_columns = ['Gender', 'Degree']

# Create a contingency table for Gender vs Degree
contingency_table = pd.crosstab(df['Gender'], df['Degree'])

# Stacked bar plot
plt.figure(figsize=(10, 6))
contingency_table.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Plot of Gender vs Degree')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Degree', bbox_to_anchor=(1, 1))
plt.show()


# In[42]:


##observation
#most of students take B.TECH./B.E. as a degree


# In[43]:


# Select categorical columns for analysis
categorical_columns = ['Gender', 'Degree', 'Specialization', 'CollegeState']

# Create a contingency table for selected categorical columns
contingency_table = pd.crosstab(index=df['Gender'], columns=df['Degree'])

# Stacked bar plot using heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap of Relationships between Categorical Columns')
plt.xlabel('Degree')
plt.ylabel('Gender')
plt.show()


# In[44]:


##observation
#Males generally taking more degrees than females


# In[45]:


df['Degree'].value_counts()


# In[46]:


filtered_df = df[df['Degree'].isin(['B.Tech/B.E.'])]

# Plot boxplot for salary comparison
plt.figure(figsize=(12, 6))
sns.boxplot(x='Degree', y='Salary', data=filtered_df)
plt.title('Comparison of Salary for B.Tech/B.E. vs Other Degrees')
plt.xlabel('Degree')
plt.ylabel('Salary')
plt.show()


# In[47]:


#observations:
#Rarely salary of B.Tech/B.E. Degree students are high


# In[48]:


# Identify the top 4 specializations
top_specializations = df['Specialization'].value_counts().nlargest(4).index

# Filter data for the top 4 specializations
filtered_df = df[df['Specialization'].isin(top_specializations)]

# Countplot for relationship between gender and top 4 specializations
plt.figure(figsize=(14, 8))
sns.countplot(x='Specialization', hue='Gender', data=filtered_df)
plt.title('Countplot of Top 4 Specializations by Gender')
plt.xlabel('Specialization')
plt.ylabel('Count')
plt.show()


# In[49]:


#Obsrrvations:
#male generally are more than female


# In[50]:


# Identify the top N specializations based on the number of occurrences
top_specializations = df['Specialization'].value_counts().nlargest(4).index

# Filter data for the top N specializations
filtered_df = df[df['Specialization'].isin(top_specializations)]

# Plot boxplot for salary comparison by specialization
plt.figure(figsize=(16, 8))
sns.boxplot(x='Specialization', y='Salary', data=filtered_df, order=top_specializations)
plt.title('Salary Comparison by Specialization')
plt.xlabel('Specialization')
plt.ylabel('Salary')
plt.show()


# In[51]:


#observation
#By seeing all boxplots we can find all Specialization have nearly same salay except computer engineering


# In[52]:


##Research Questions


# In[53]:


df['Designation'].value_counts()


# In[54]:


df[df['Designation']=='programming analyst']


# In[55]:


df['Designation'].unique()


# In[56]:


df[df['Designation']=='hardware engineer']


# In[57]:


df[df['Designation']=='hardware engineer']


# In[58]:


# Filter designations containing the term 'Analyst'
analyst_designations = df[df['Designation'].str.contains('fresher', case=False, na=False)]

# Display the counts
print("Designation Counts with 'Analyst':")
print(analyst_designations['Designation'].value_counts())


# In[59]:


# Filter designations containing the term 'Analyst'
analyst_designations = df[df['Designation'].str.contains('Analyst', case=False, na=False)]

# Display the counts
print("Designation Counts with 'Analyst':")
print(analyst_designations['Designation'].value_counts())


# In[60]:


# Choose specific designations for analysis
selected_designations = ['programmer analyst', 'software engineer', 'hardware engineer', 'associate engineer']

# Filter the data for selected designations
filtered_df = df[df['Designation'].isin(selected_designations)]

# Convert Salary to lakhs for selected designations
filtered_df['Salary_in_lakhs'] = filtered_df['Salary'] / 100000

# Boxplot for salary comparison by gender and designation
plt.figure(figsize=(14, 8))
sns.boxplot(x='Designation', y='Salary_in_lakhs', hue='Gender', data=filtered_df)
plt.title('Salary Comparison by Gender and Designation (in Lakhs)')
plt.xlabel('Designation')
plt.ylabel('Salary (in Lakhs)')
plt.show()


# In[61]:


df['Specialization'].value_counts()


# In[62]:


# Filter the data for Computer Science Engineering graduates with selected specializations and designations
cs_graduates_df = df[(df['Specialization'] == 'computer engineering') &
                     (df['Designation']=='hardware engineer') ]


cs_graduates_df


# In[63]:


#insight:
#there is no rows for designation= hardware engineer and specialization= computer science and engineering


# In[64]:


# Choose specific designations for analysis
selected_designations = ['programmer analyst', 'software engineer', 'hardware engineer', 'associate engineer']

# Filter the data for Computer Science Engineering graduates with selected specializations and designations
cs_graduates_df = df[(df['Specialization'] == 'computer science & engineering') &
                     (df['Designation'].isin(selected_designations)) &
                     (df['Degree'] == 'B.Tech/B.E.')]

# Convert Salary to lakhs for Computer Science Engineering graduates with selected designations
cs_graduates_df['Salary_in_lakhs'] = cs_graduates_df['Salary'] / 100000

# Boxplot for salary comparison by designation for Computer Science Engineering graduates
plt.figure(figsize=(14, 8))
sns.boxplot(x='Designation', y='Salary_in_lakhs', data=cs_graduates_df)
plt.title('Salary Comparison by Designation for Computer Science Engineering Graduates (in Lakhs)')
plt.xlabel('Designation')
plt.ylabel('Salary (in Lakhs)')
plt.show()


# In[65]:


#Observation
#Looking at above plot It completely supports the claim of Times of India article dated Jan 18, 2019 states that “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.”
#only the issue is I have no data of designation= hardware engineer and specialization= computer science and engineering


# In[66]:


# Countplot for the relationship between gender and specialization
plt.figure(figsize=(14, 8))
sns.countplot(x='Specialization', hue='Gender', data=df)
plt.title('Relationship between Gender and Specialization')
plt.xlabel('Specialization')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[ ]:


observations:
By looking at above plot we can see males are doing more specialization than females

