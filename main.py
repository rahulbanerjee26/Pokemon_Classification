import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , accuracy_score, precision_score, recall_score

def title(s):
    st.text("")
    st.title(s)
    st.text("")

def clean_and_split(df):
    legendary_df = df[df['is_legendary'] == 1]
    normal_df = df[df['is_legendary'] == 0].sample(75)
    legendary_df.fillna(legendary_df.mean(),inplace=True)
    normal_df.fillna(normal_df.mean(),inplace=True)
    feature_list = ['weight_kg' , 'height_m' , 'sp_attack' , 'attack' , 'sp_defense' , 'defense' , 'speed' , 'hp' ,     'is_legendary']
    sub_df = pd.concat([legendary_df,normal_df])[feature_list]
    X = sub_df.loc[:, sub_df.columns != 'is_legendary']
    Y = sub_df['is_legendary']
    X_train, X_test , y_train , y_test = train_test_split(X ,Y ,random_state=1 ,test_size= 0.2 ,shuffle=True,stratify=Y)
    return X_train , X_test , y_train , y_test


# Intro
st.title('Is that a Legendary Pokemon?')
st.image('bg.jpg', width=600)
st.markdown('''
Photo by [Kamil S](https://unsplash.com/@16bitspixelz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/pokemon?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)
''')

# Load Data
df = pd.read_csv('pokemon.csv')
st.dataframe(df.head())

# Basic Info
shape = df.shape
num_total = len(df)
num_legendary = len(df[df['is_legendary'] == 1])
num_non_legendary = num_total - num_legendary

st.subheader('''
Number of Pokemons: {}
'''.format(num_total))

st.subheader('''
Number of Legendary Pokemons: {}
'''.format(num_legendary))

st.subheader('''
Number of Non Legendary Pokemons: {}
'''.format(num_non_legendary))

st.subheader('''
Number of Features :{}
'''.format(shape[1]))

title('Legendary Pokemon Distribution based on Type')
legendary_df = df[df['is_legendary'] == 1]
fig1 = plt.figure()
ax = sns.countplot(data=legendary_df , x = 'type1',order=legendary_df['type1'].value_counts().index)
plt.xticks(rotation=45)
st.pyplot(fig1)
st.markdown("### Most of the Legnendary pokemons are Psychic Type. In fact, the number of Psychic Type Legendary Pokemons are greater than the sum of the next two most common types(Dragon and Water)")

title('Height vs Weight for Legendary and Non-Legendary Pokemons')
fig2 = plt.figure()
sns.scatterplot(data=df , x = 'weight_kg' , y = 'height_m' , hue='is_legendary')
st.pyplot(fig2)
st.markdown('### While there are a few outliers, most of the Non-Legendary Pokemons are clustered towards the lower left corner of the graph')
st.markdown('### If a pokemon is heavier than 600 kg, it is most likely a legendary pokemon')

title('Correlation between features')
fig3 = plt.figure()
sns.heatmap(legendary_df[['attack','sp_attack','defense','sp_defense','height_m','weight_kg','speed']].corr())
st.pyplot(fig3)

title('Special Attack vs Attack')
fig4 = plt.figure()
sns.scatterplot(data=df, x='sp_attack',y='attack',hue='is_legendary')
st.pyplot(fig4)


title('Random Forest')
X_train , X_test , y_train , y_test = clean_and_split(df)
st.subheader("Sample Data")
st.dataframe(X_train.head(3))
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train , y_train)

title("Metrics")
st.subheader("Model Score: {}".format(model.score(X_test , y_test)))
st.subheader("Precision Score: {}".format(precision_score(model.predict(X_test) , y_test)))
st.subheader("Recall Score: {}".format(recall_score(model.predict(X_test) , y_test)))

st.subheader("Confusion Matrix")
fig5 = plt.figure()
conf_matrix = confusion_matrix(model.predict(X_test) , y_test)
sns.heatmap(conf_matrix , annot=True , xticklabels=['Normal' , 'Legendary'] , yticklabels=['Normal' , 'Legendary'])
plt.ylabel("True")
plt.xlabel("Predicted")
st.pyplot(fig5)