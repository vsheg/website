'''Flask app entry point.'''


from functools import partial
from os import getenv
from flask import Flask, redirect, render_template
from models import contacts, education, experience
from markdown import markdown

render_template = partial(render_template, repo_url=getenv('REPO_URL'), website_url=getenv('WEBSITE_URL'))

app = Flask(__name__)
app.jinja_env.add_extension('pypugjs.ext.jinja.PyPugJSExtension')


@app.context_processor
def markdown_processor():
    '''Add `markdown(md_text)` to the template processor.'''

    # inner function
    def markdown_(text: str) -> str:
        '''Render markdown and trim terminal `<p>` tags.'''
        return markdown(text).removeprefix('<p>').removesuffix('</p>')  # TODO: switch to a nicer markdown processor

    return dict(markdown=markdown_)


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
