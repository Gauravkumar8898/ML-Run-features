from flask import Flask, request, jsonify,make_response
import mlrun
from transformers import pipeline

app = Flask(__name__)

class Mlrun_Pipeline_Transformer:
    def __init__(self):
        self.project = mlrun.get_or_create_project("tensorflow-mlrun", "./", user_project=True, init_git=True)
        self.context = mlrun.get_or_create_ctx("transformers-example")
        self.text_pipeline = pipeline("sentiment-analysis")

    @mlrun.handler()
    def predict_sentiment(self, texts):
        results = []
        for text in texts:
            result = self.text_pipeline(text)
            results.append({
                "text": text,
                "sentiment": result[0]['label'],
                "confidence": result[0]['score']
            })
            self.context.log_result(key=f"Text: {text}",
                                    value=f"Sentiment: {result[0]['label']} with confidence: {result[0]['score']}")
        return results


@app.route('/predict', methods=['POST'])
def predict():
    obj = Mlrun_Pipeline_Transformer()
    if request.method == 'POST':
        data = request.get_json()
        texts = data.get('text')
        results = obj.predict_sentiment(texts)
        print(results)
        obj.project.context.log_result(results)
        return make_response(jsonify(results),200)



app.run(port=8004,debug=True)

