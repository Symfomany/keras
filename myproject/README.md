<!-- Run with debug -->

### Quickstart
```
    flask run
```

Request Object Post, Cookie, Upload
http://flask.pocoo.org/docs/1.0/quickstart/#the-request-object

GET 

searchword = request.args.get('key', '')

* request.method == 'POST'
* request.path == '/hello'
* request.form['username']
* request.files['the_file']
* f.save('/var/www/uploads/uploaded_file.txt')
* return redirect(url_for('login'))
* abort(401) et this_is_never_executed()
* if 'username' in session: return 'Logged in as %s' % escape(session['username'])
* session['username'] = request.form['username']
* app.logger.debug('A value for debugging')


Deploiement

http://flask.pocoo.org/docs/1.0/deploying/#deployment


Awesome Flask
https://github.com/humiaozuzu/awesome-flask