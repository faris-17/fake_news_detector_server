import warnings
from fastapi import FastAPI, Request, HTTPException
from scipy.sparse import csr_matrix
from fastapi.encoders import jsonable_encoder
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from datetime import date

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Download necessary NLTK corpora
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = FastAPI()

origins = ["*"]

# Custom unpickler to handle deprecated imports
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "scipy.sparse.csr" and name == "csr_matrix":
            return csr_matrix
        return super().find_class(module, name)

# Load the model and vectorizer
try:
    with open('C:/fake_news_detector/datasets/model.pkl', 'rb') as f:
        prediction_model = CustomUnpickler(f).load()

    with open('C:/fake_news_detector/datasets/tfidf.pkl', 'rb') as f:
        tfidf_v = CustomUnpickler(f).load()
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found: {e}")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection string
MONGO_URI = "mongodb+srv://faris:<12345>@cluster0.9gog3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
database = client.newsset
news_collection = database.get_collection("news")


class Output:
    def __init__(self, news='Please provide a headline', prediction='None', status='500 Internal Server Error'):
        self.news = news
        self.prediction = prediction
        self.status = status

@app.get("/status")
async def status():
    return {"status": "200 OK"}

@app.get("/api/predict")
async def predict_news(request: Request):
    try:
        news = request.query_params.get('news', None)
        if not news:
            raise HTTPException(status_code=400, detail="Query parameter 'news' is required.")
        return fake_news_det(news)
    except Exception as e:
        return Output(status='500 Internal Server Error', prediction='None', news=str(e)).__dict__

@app.get("/api/predictednews")
async def get_predicted_news():
    all_news = []
    async for news in news_collection.find():
        all_news.append(newsHelper(news))
    return all_news

@app.post("/api/sendnews")
async def send_news(request: Request):
    json_body = await request.json()
    if "title" not in json_body or "source" not in json_body:
        raise HTTPException(status_code=400, detail="Fields 'title' and 'source' are required.")
    
    prediction = fake_news_det(json_body["title"])
    today = date.today()
    date_posted = today.strftime("%B %d, %Y")
    data = {
        "title": json_body["title"],
        "source": json_body["source"],
        "date_posted": date_posted,
        "prediction": prediction["prediction"]
    }
    await news_collection.insert_one(jsonable_encoder(data))
    return data

def newsHelper(news) -> dict:
    return {
        "id": str(news["_id"]),
        "title": news["title"],
        "source": news["source"],
        "date_posted": news["date_posted"],
        "prediction": news["prediction"],
    }

def fake_news_det(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    tokens = word_tokenize(review)  # Tokenize the news headline
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    input_data = [' '.join(processed_tokens)]  # Join processed tokens into a single string
    vectorized_input_data = tfidf_v.transform(input_data)  # Transform using TF-IDF
    prediction = prediction_model.predict(vectorized_input_data)  # Predict using the trained model
    
    if prediction[0] == 1:
        return {"news": news, "prediction": "Fake", "status": "200 OK"}
    else:
        return {"news": news, "prediction": "Real", "status": "200 OK"}
