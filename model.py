import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#sample data
data = {
    "message":[
        "Win money now",
        "Hello how are you",
        "Claim your free prize",
        "Let's meet tomorrow",
        "Congratulations you won lottery"
    ],
    "label":[1,0,1,0,1]
    
}

df = pd.DataFrame(data)

x = df["message"]
y = df["label"]

#convert text to numbers

vectorizer = TfidfVectorizer()

x_vec = vectorizer.fit_transform(x)

#train model

model = MultinomialNB()
model.fit(x_vec, y)

#save model

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer,open("vectorizer.pkl", "wb"))

print ("model trained and saved")



