from fastapi import FastAPI
import pickle

app = FastAPI()


#load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.get("/")
def home():
    return {"message":"spam detection api runing"}

@app.post("/predict")
def predict(message: str):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return{
        "message":message,
        "prediction":result
        }
    