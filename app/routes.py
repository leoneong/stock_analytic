from flask import render_template, flash, redirect, url_for, request
from app import app, db
from app.forms import SearchForm
from app.models import Stock
import xlrd 


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
#     path = "data/svm_testing.xlsx" #downPath承接上面下载文件的路径，这个读取文件路径都是可以换成自己的
#     workbook = xlrd.open_workbook(path)  #打开excel文件

#     Data_sheet = workbook.sheets()[0]  # 通过索引获取
#     rowNum = Data_sheet.nrows  # sheet行数
#     colNum = Data_sheet.ncols  # sheet列数
  

#     list = []
    
#     for i in range(rowNum):
#         rowlist = []
#         for j in range(colNum):
#             rowlist.append(Data_sheet.cell_value(i, j))
#         list.append(rowlist)
    
#     del list[0] #删掉第一行，第一行获取的是文件的头，一般不用插到数据库里面
    
#   #  接下来是把数据插到数据库里面,以下是我自己的数据库，大家可以根据自己的需要自行处理
#     print(Data_sheet.cell_value(1, 2))

#     for a in list:   
#         stock = Stock()
#         stock.stockname= a[0]
#         stock.trend = a[1]
#         stock.accuracy = a[2]
#         stock.risk = a[3]
#         db.session.add(stock)
#         db.session.commit()    



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