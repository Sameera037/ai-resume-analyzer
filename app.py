from flask import Flask, render_template, request
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

nlp = spacy.load('en_core_web_sm')

def extract_skills(text):
    doc = nlp(text.lower())
    skills = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return ' '.join(skills)

def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    if request.method == 'POST':
        resume_text = request.form['resume']
        jd_text = request.form['job_description']
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        score = round(calculate_similarity(resume_skills, jd_skills)*100, 2)
    return render_template('index.html', score=score)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)