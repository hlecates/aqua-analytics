from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from .session import Base

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    gender = Column(String(8), nullable=False)
    distance = Column(Integer, nullable=False)
    stroke = Column(String(32), nullable=False)
    is_relay = Column(Boolean, nullable=False, default=False)
    __table_args__ = (UniqueConstraint("gender", "distance", "stroke", "is_relay", name="uq_event_def"),)
    results = relationship("ResultIndividual", back_populates="event")

class Meet(Base):
    __tablename__ = "meets"
    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    results = relationship("ResultIndividual", back_populates="meet")

class School(Base):
    __tablename__ = "schools"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    athletes = relationship("Athlete", back_populates="school")

class Athlete(Base):
    __tablename__ = "athletes"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False, index=True)
    grad_year = Column(Integer, nullable=True)
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=True)
    school = relationship("School", back_populates="athletes")
    results = relationship("ResultIndividual", back_populates="athlete")

class ResultIndividual(Base):
    __tablename__ = "results_individual"
    id = Column(Integer, primary_key=True)
    meet_id = Column(Integer, ForeignKey("meets.id"), nullable=False, index=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False, index=True)
    athlete_id = Column(Integer, ForeignKey("athletes.id"), nullable=True)
    athlete_name = Column(String(128), nullable=False)
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=True)
    school_name = Column(String(128), nullable=True)
    round = Column(String(32), nullable=True)           # 'Prelim' or 'Final'
    place = Column(Integer, nullable=True)
    time_seconds = Column(Float, nullable=True, index=True)
    time_raw = Column(String(32), nullable=True)
    __table_args__ = (
        UniqueConstraint("meet_id", "event_id", "athlete_name", "time_seconds", "round", name="uq_result_identity"),
        Index("ix_meet_event", "meet_id", "event_id"),
    )
    meet = relationship("Meet", back_populates="results")
    event = relationship("Event", back_populates="results")
    athlete = relationship("Athlete", back_populates="results")
    school = relationship("School")
    
