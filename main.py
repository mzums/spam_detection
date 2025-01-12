import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# plt.figure(figsize=(6, 4))
# sns.countplot(x='label', data=df, palette='cool')
# plt.title("Proportions of ham and spam messages")
# plt.xticks([0, 1], ['Ham', 'Spam'])
# plt.ylabel('Number of messages')
# plt.xlabel('Label')
# plt.show()
# 
# df['message_length'] = df['message'].apply(len)
# 
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, palette='cool', alpha=0.7)
# plt.title("Message length distribution for Ham and Spam")
# plt.xlabel('Message length')
# plt.ylabel('Number of messages')
# plt.legend(['Ham', 'Spam'])
# plt.show()

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_message(message):
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)
    return "spam" if prediction[0] == 1 else "ham"

custom_message = "Hi! Text me later."
print(f"Message: '{custom_message}' is classified as: {predict_message(custom_message)}")
