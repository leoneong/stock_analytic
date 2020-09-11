from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class SearchForm(FlaskForm):
    stockid = StringField('STOCK ID', validators = [DataRequired()])
    submit = SubmitField('Search')

