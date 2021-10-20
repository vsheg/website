'''Flask app entry point.'''


from functools import partial
from os import getenv

from flask import Flask, redirect, render_template
from flask_assets import Bundle, Environment

from models import contacts, education, experience

render_template = partial(
    render_template, repo_url=getenv('REPO_URL'), website_url=getenv('WEBSITE_URL')
)

app = Flask(__name__)
app.jinja_env.add_extension('pypugjs.ext.jinja.PyPugJSExtension')


assets = Environment(app)
assets.init_app(app)
all_css = Bundle(
    'styles/*.scss',
    filters='pyscss,cssmin',
    depends='styles/*.scss',
    output='styles/all_sass.css',
)
assets.register(None, all_css)
all_css.build()


@app.context_processor
def add_envs():
    '''Add environment variables to app context.'''
    return {'REPO_URL': getenv('REPO_URL'), 'WEBSITE_URL': getenv('WEBSITE_URL')}


@app.route('/')
def main_page():
    '''Redirects to `/contacts`.'''
    return redirect('/contacts')


@app.route('/contacts')
def contacts_page():
    '''Returns Contacts page.'''
    return render_template('contacts.pug', title='Contacts', contacts=contacts)


@app.route('/cv')
def cv_page():
    '''Returns CV page.'''
    return render_template('cv.pug', title='CV', experience=experience, education=education)
