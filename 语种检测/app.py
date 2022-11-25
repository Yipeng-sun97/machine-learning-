from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load


class LanguageDetector():
    # 成员函数
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000, preprocessor=self._remove_noise)

    # 私有函数，数据清洗
    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text

    # 特征构建
    def features(self, X):
        return self.vectorizer.transform(X)

    # 拟合数据
    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    # 预估类别
    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    # 测试集评分
    def score(self, X, y):
        return self.classifier.score(self.features(X), y)
    
    # 模型持久化存储
    def save_model(self, path):
        dump((self.classifier, self.vectorizer), path)
    
    # 模型加载
    def load_model(self, path):
        self.classifier, self.vectorizer = load(path)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	model_path = "model/language_detector.model"
	my_language_detector = LanguageDetector()
	my_language_detector.load_model(model_path)

	if request.method == 'POST':
		message = request.form['message']
		my_prediction = my_language_detector.predict(message)
	return render_template('result.html',prediction = my_prediction[0])



if __name__ == '__main__':
	app.run(debug=True)