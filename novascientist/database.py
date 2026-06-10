import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text, ForeignKey, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class GoalDirectory(Base):
    __tablename__ = 'goal_directories'
    
    hash_id = Column(String, primary_key=True)
    goal_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemState(Base):
    __tablename__ = 'system_states'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    goal_hash = Column(String, ForeignKey('goal_directories.hash_id'))
    iteration = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    literature_review = Column(JSON, nullable=True)
    meta_reviews = Column(JSON, default=list)
    supervisor_decisions = Column(JSON, default=list)
    final_report = Column(JSON, nullable=True)
    actions = Column(JSON, default=list)
    cosine_similarity_trajectory = Column(JSON, default=list)
    cluster_count_trajectory = Column(JSON, default=list)

class HypothesisModel(Base):
    __tablename__ = 'hypotheses'
    
    uid = Column(String, primary_key=True)
    goal_hash = Column(String, ForeignKey('goal_directories.hash_id'))
    location = Column(String, nullable=False) # 'generated', 'reviewed', 'evolved', 'reflection_queue', 'tournament'
    
    hypothesis_text = Column(Text, nullable=False)
    predictions = Column(JSON, default=list)
    assumptions = Column(JSON, default=list)
    parent_uid = Column(String, nullable=True)
    
    # Fields for ReviewedHypothesis
    causal_reasoning = Column(Text, nullable=True)
    assumption_research_results = Column(JSON, nullable=True)
    verification_result = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

def get_engine(db_path="sqlite:///novascientist.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
