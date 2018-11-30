from flask import Flask, url_for, render_template
import tensorflow as tf
from ../load import load_graph
from flask_cors import CORS

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
    pass

@app.route('/julien')
def hello():
    return 'Hello, julien'
    pass

@app.route('/messieurs/<name>')
def hi(name=None):
    return render_template('messieurs.html', name=name)
    pass

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return do_the_login()
    else:
        return show_the_login_form()
        
# PARAM
@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username
    pass

# post id
@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id
    pass