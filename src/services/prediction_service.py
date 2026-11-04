"""
File: prediction_service.py
Project: Intelligent Fruit Classification and Information System
Description:
    Coordinates the inference workflow by connecting image
    preprocessing, model inference, prediction policy, fruit
    information retrieval, and prediction-history persistence.

Author(s): Group 8
Integration Lead: Hasnain Younas
ML Contributor: Shubhan Singh
Database Contributor: Abeeha Mariam
Responsible Area: Shared/Core Integration
Maintainers: Hasnain Younas, Shubhan Singh, Abeeha Mariam
"""

# the code for this file is in the hasnain folder so it is call from there
from hasnain.prediction_service import PredictionService, get_history, get_summary, HISTORY_NOT_SAVED, FRUIT_INFO_MISSING

ISSUE_HISTORY_NOT_SAVED = HISTORY_NOT_SAVED
ISSUE_FRUIT_INFO_UNAVAILABLE = FRUIT_INFO_MISSING
