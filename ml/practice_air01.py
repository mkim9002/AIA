import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# 감정 분석을 위한 모델 초기화
sia = SentimentIntensityAnalyzer()

# 감정 분석 수행할 문장
sentence = "I love this product so much!"

# 문장에 대한 감정 분석 결과 출력
print(sia.polarity_scores(sentence))
