from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    userSignedIn = False
    if userSignedIn:
        user = {'username': 'Christopher'}
        return render_template('index.html', title='Home', user=user)
    else:
        return render_template('index.html', title='Home')
