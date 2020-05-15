from app import db

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stockname = db.Column(db.String(64), index=True, unique=True)
    description = db.Column(db.String(120))
    daytrend = db.Column(db.String(64))
    monthtrend = db.Column(db.String(64))


    def __repr__(self):
        return '<Stock {}>'.format(self.stockname)   

class Index(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    indexname = db.Column(db.String(64), index=True, unique=True)
    daytrend = db.Column(db.String(64))
    monthtrend = db.Column(db.String(64))


    def __repr__(self):
        return '<Stock {}>'.format(self.stockname)   