from flask import render_template, flash, redirect, url_for, request
from app import app, db
from app.forms import SearchForm
from app.models import Stock, Index


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():

    page = request.args.get('page', 1, type=int)
    stocks = Stock.query.order_by(Stock.id).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('index', page=stocks.next_num) \
        if stocks.has_next else None
    prev_url = url_for('index', page=stocks.prev_num) \
        if stocks.has_prev else None

    form = SearchForm()
    if form.validate_on_submit():
        stocks = Stock.query.filter_by(stockname=form.stockid.data).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
        next_url = url_for('index', page=stocks.next_num) \
            if stocks.has_next else None
        prev_url = url_for('index', page=stocks.prev_num) \
            if stocks.has_prev else None
        redirect(url_for('index'))
    
    
    return render_template('index.html', title = 'Home', stocks = stocks.items, form = form, next_url=next_url,
                           prev_url=prev_url )

    
@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'Stock': Stock}