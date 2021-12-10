from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, types
from sqlalchemy.orm import relationship

import zlib
import numpy as np

from db_session import Base, SessionLocal


class Test(Base):
    __tablename__ = "tests"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(20), nullable=True, unique=True)
    n_ins = Column(Integer, nullable=False)
    n_objs = Column(Integer, nullable=False)
    n_cons = Column(Integer, nullable=False)

    kernel_id = Column(Integer, ForeignKey("kernels.id"), nullable=False,)
    acq_id = Column(Integer, nullable=False)

class Kernel(Base):
    __tablename__ = "kernels"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(30))

class AcqFunction(Base):
    __tablename__ = "acqfunctions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(10), nullable=False, unique=True)

class Input(Base):
    __tablename__ = "inputs"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(20))
    test_id = Column(Integer, ForeignKey("tests.id"))

class Output(Base):
    __tablename__ = "outputs"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(20))
    test_id = Column(Integer, ForeignKey("tests.id"))

class NumpyType (types.TypeDecorator):
    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        return zlib.compress(value.dumps(), 9)

    def process_result_value(self, value, dialect):
        return np.loads(zlib.decompress(value))

if __name__=="__main__":

    db = SessionLocal()

    MES = AcqFunction(name="MES")
    MESMO = AcqFunction(name="MESMO")
    
    RBF = Kernel(name="RBF")

    db.add(MES)
    db.add(MESMO)
    db.add(RBF)

    db.commit()


