from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

class TextBaselineModel:
    """
    Unimodal text baseline classifier using TF-IDF feature extraction
    and Logistic Regression.
    """
    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        
    def fit(self, texts, labels):
        """
        Fits the TF-IDF vectorizer and the Logistic Regression classifier.
        """
        features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(features, labels)
        
    def predict(self, texts):
        """
        Predicts labels for a list of texts.
        """
        features = self.vectorizer.transform(texts)
        return self.classifier.predict(features)
        
    def predict_proba(self, texts):
        """
        Predicts label probabilities for a list of texts.
        """
        features = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(features)
        
    def save(self, model_path: str):
        """
        Saves the complete pipeline to disk.
        """
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, model_path)
        
    def load(self, model_path: str):
        """
        Loads the pipeline from disk.
        """
        saved_dict = joblib.load(model_path)
        self.vectorizer = saved_dict["vectorizer"]
        self.classifier = saved_dict["classifier"]
