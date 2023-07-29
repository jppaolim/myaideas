from flask import Flask, request, render_template
from myaideas import main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = main(query)
        return render_template('index.html', results=results)
    return render_template('index.html')
if __name__ == "__main__":
    app.run()
    app.run(host='0.0.0.0', port=8000)
