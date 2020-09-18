from app import db

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stockname = db.Column(db.String(64), index=True, unique=True)
    trend = db.Column(db.Integer)
    accuracy = db.Column(db.Integer)
    risk = db.Column(db.Integer)

    def __repr__(self):
        return '<Stock {}>'.format(self.stockname)   


