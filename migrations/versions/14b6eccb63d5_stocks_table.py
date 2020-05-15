"""stocks table

Revision ID: 14b6eccb63d5
Revises: 
Create Date: 2020-05-02 15:09:08.210120

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '14b6eccb63d5'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('stock',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('stockname', sa.String(length=64), nullable=True),
    sa.Column('description', sa.String(length=120), nullable=True),
    sa.Column('daytrend', sa.String(length=64), nullable=True),
    sa.Column('monthtrend', sa.String(length=64), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('description')
    )
    op.create_index(op.f('ix_stock_stockname'), 'stock', ['stockname'], unique=True)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_stock_stockname'), table_name='stock')
    op.drop_table('stock')
    # ### end Alembic commands ###
