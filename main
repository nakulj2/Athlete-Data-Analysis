import pandas as pd
import seaborn as sns
import missingno as msno
import plotly.express as px
import numpy as np

df = pd.read_csv("data.xlsx", low_memory=False)

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df.head()
df.shape
df.info()
df.isnull().sum()
df.nunique()

df.drop(['Height', 'Weight', 'Arm Relaxed Circumfarance', 'Forearm Circumfarance', 'Chest Circumfarance', 'Waist Circumfarance', 'Glutal Circumfarance', 'Up thigh Circumfarance'], axis=1, inplace=True)

df = df.drop_duplicates()
df.info()

percentage_null = df.isnull().sum() / df.shape[0] * 100
percentage_null = pd.DataFrame({"columns": percentage_null.keys(), "%": percentage_null})
percentage_null.reset_index(drop=True, inplace=True)
msno.bar(df)


df.columns = df.columns.str.replace(' ', '_')


categorical_values = df.select_dtypes(include=[object])

numerical_values = df.select_dtypes(include=[np.float64, np.int64])


categorical_values.head()


numerical_values.head()

categorical_values.isnull().sum()

sns.countplot(x='host_identity_verified', data=df)
data = df['host_identity_verified'].value_counts()
data.plot(kind='pie', autopct='%0.1f%%')

df.fillna(0, inplace=True)

sns.countplot(y='neighbourhood_group', data=df)

df['neighbourhood_group'].value_counts()


df['Wrist Circumference'].nunique()

neighbourhood_count = df['neighbourhood'].value_counts()
top_15 = neighbourhood_count.head(15)


top_15.plot(kind='bar', figsize=(10, 5))
plt.xlabel('Wrist Circumference')
plt.ylabel('BMI')
plt.title('Wrist Circumference vs BMI')
plt.show()

df['Percentile'].nunique()
df['Percentile'].value_counts()
mode = df['Percentile'].mode().iloc[0]
df['Percentile'].fillna(mode, inplace=True)
df['Age'].isnull().sum()
df['Age'].value_counts()
mean = df['Age'].mode().iloc[0]
df['Age'].fillna(mean, inplace=True)

data.plot(kind='pie', autopct='%0.1f%%')

df['tmp'] = 1
fig = px.pie(df, names='instant_bookable', values='tmp', hole=0.6, title="instant_bookable")
fig.update_traces(textposition='outside', textinfo='percent+label')
fig.update_layout(title_text='instant_bookable',
                  annotations=[dict(text='instant_bookable', x=0.5, y=0.5, font_size=10, showarrow=False)])

sns.countplot(x='Gender', data=df)


df['Gender'].value_counts()
df['Gender'].fillna('Male', inplace=True)


sns.countplot(x='room_type', data=df)

df['type'].value_counts()

df['type'].fillna(' ', inplace=True)
def remove_percentage_sign(value):
    if pd.isna(value):
        return np.NaN
    else:
        return float(value.replace("$", "").replace(",", "").replace(" ", ""))


df['Total'] = df['Total'].apply(lambda x: remove_percentage_sign(x))
df['Total'] = df['Total'].apply(lambda x: remove_percentage_sign(x))


plt.figure(figsize=(15, 10))
plt.title("Relationship between Total and Percentile")
sns.scatterplot(x=df.price, y=df.service_fee, hue=df.room_type, s=30);

def get_year(date):
    try:
        return str(date).split("/")[2]
    except:
        pass


df['Testing Date'] = df['Testing Date'].apply(get_year)


fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(y='Testing Date', data=df, ax=ax)

df['Testing Date'].median()

df['Testing Date'].fillna(2016, inplace=True)

df['Testing Date'].isnull().sum()
df.isnull().sum()

year = df['Ratio'].value_counts()
plt.figure(figsize=(20, 8))
sns.pointplot(x=year.index, y=year.values)
plt.xlabel("Ratio")
plt.ylabel("Percentile")
plt.title("Ratio vs Percentile")
df['Percentile'].isnull().sum()

mode = df['Percentile'].mode().iloc[0]

df['Percentile'].fillna(2014, inplace=True)

fig = px.histogram(df, x='Percentile')
fig.show()
numerical_values.isnull().sum()

df.isnull().sum()

df.head()
df = df.dropna()
df.isnull().sum()


df.corr()

df.drop('tmp', axis=1, inplace=True)

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between columns')
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('preprocessed_data.xlsx')  # Replace 'data.xlsx' with the actual file path

# Step 2: Preprocess the data
X = data.drop('Wrist Circumfarance', axis=1)  # Replace 'target_column' with the actual target column name
y = data['Wrist Circumfarance']  # Replace 'target_column' with the actual target column name

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features (optional but recommended for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
