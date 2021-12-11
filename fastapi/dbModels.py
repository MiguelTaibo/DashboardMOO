from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, types, Float

from db_session import Base, SessionLocal

class Test(Base):
    __tablename__ = "tests"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(20), nullable=True, unique=True)
    n_ins = Column(Integer, nullable=False)
    n_objs = Column(Integer, nullable=False)
    n_cons = Column(Integer, nullable=False)
    kernel_id = Column(Integer, nullable=False)
    acq_id = Column(Integer, nullable=False)
    acq_M = Column(Integer, nullable=True)
    acq_N = Column(Integer, nullable=True)

class Input(Base):
    __tablename__ = "inputs"
    id = Column(Integer, primary_key=True, index=True)
    lowerBound = Column(Float)
    upperBound = Column(Float)
    name = Column(String(20))
    test_id = Column(Integer, ForeignKey("tests.id"))

class Output(Base):
    __tablename__ = "outputs"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(20))
    maximize = Column(Boolean)
    test_id = Column(Integer, ForeignKey("tests.id"))
