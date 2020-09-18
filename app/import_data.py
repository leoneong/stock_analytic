from xlrd import open_workbook
from app import db
from app.models import Stock


path = "data/svm.testing.xlsx" #downPath承接上面下载文件的路径，这个读取文件路径都是可以换成自己的
workbook = xlrd.open_workbook(path)  #打开excel文件

Data_sheet = workbook.sheets()[0]  # 通过索引获取
rowNum = Data_sheet.nrows  # sheet行数
colNum = Data_sheet.ncols  # sheet列数

list = []
for i in range(rowNum):
    rowlist = []
    for j in range(colNum):
        rowlist.append(Data_sheet.cell_value(i, j))
    list.append(rowlist)
    del list[0] #删掉第一行，第一行获取的是文件的头，一般不用插到数据库里面
    
  #  接下来是把数据插到数据库里面,以下是我自己的数据库，大家可以根据自己的需要自行处理

for a in list:
    stock = Stock()
    stock.stockname= a[0]
    stock.trend = a[1]
    stock.accuracy = a[2]
    stock.risk = a[3]
        
db.session.add(stock)
db.session.commit()    
print(list)