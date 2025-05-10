from flask import Flask, render_template, request
import model

app = Flask(_name_)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    size = float(request.form['size'])
    bedrooms = int(request.form['bedrooms'])
    location = request.form['location']

    prediction = model.predict_price(size, bedrooms, location)
    return render_template('index.html', price=round(prediction, 2))

if _name_ == '_main_':
    app.run(debug=True)