from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os

user_name = os.environ.get("SQL_USER", "miguel")
password = os.environ.get("SQL_PASSWORD", "pass")
host = os.environ.get("SQL_HOST", "db")
database_name = os.environ.get("SQL_DATABASE", "MOOdb")


DATABASE = 'mysql://%s:%s@%s/%s' % (
    user_name,
    password,
    host,
    database_name,
)

print(DATABASE)

SQLALCHEMY_DATABASE_URL = "mysql://root@127.0.0.1:3306/TFM-pruebas"

engine = create_engine(
    DATABASE, pool_recycle=3306
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()