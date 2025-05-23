from flask import Flask, render_template, request
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# โหลดโมเดลและข้อมูล
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = torch.load("export_course_model/course_embeddings.pt")
df = pd.read_csv("export_course_model/course_data.csv")

# ฟังก์ชันแนะนำคอร์ส
def recommend(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)
    recommendations = []
    for score, idx in zip(top_results.values, top_results.indices):
        course = df.iloc[idx.item()]
        recommendations.append({
            'name': course['Course Name'],
            'university': course['University'],
            'level': course['Difficulty Level'],
            'rating': course['Course Rating'],
            'url': course['Course URL']
        })
    return recommendations

# เริ่มต้น Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    courses = []
    if request.method == "POST":
        query = request.form.get("query")
        courses = recommend(query)
    return render_template("index.html", courses=courses)

if __name__ == "__main__":
    app.run(debug=True)
