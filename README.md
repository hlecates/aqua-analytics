# Aqua Analytics

A comprehensive machine learning project for predicting competitive swimming performance across multiple levels of competition. Aqua Analytics applies data science techniques to analyze swimming meet data and forecast various aspects of race outcomes, from individual performance metrics to competitive dynamics.

## Project Overview

Aqua Analytics operates at two distinct competitive levels:

- **National Level**: A proof of concept modeling framework for elite national swimming competitions, focusing on world record residuals, American record differentials, and top seed win probabilities. The models achieve R² > 0.9 for time predictions, demonstrating the effectiveness of advanced feature engineering and machine learning techniques in swimming performance analysis.
- **NESCAC Level**: A complete web application for Division III conference competition within the New England Small College Athletic Conference, featuring a React frontend, Flask backend, and comprehensive data pipeline for predicting finals qualification and scoring opportunities.

The project combines web scraping, feature engineering, machine learning, and statistical analysis to provide insights into swimming performance patterns and competitive outcomes.

## Project Structure

### `national/`
A proof of concept modeling framework for national-level swimming competitions. This includes data collection from major meets (Olympics, World Championships, National Championships), comprehensive feature engineering of competitive dynamics (field depth, record proximity, swimmer demographics), and production-ready machine learning models that predict time differentials from world records and classify race outcomes. The models achieve R² > 0.9 for time predictions, demonstrating the viability of machine learning approaches in elite swimming performance analysis.

### `nescac/`
A complete web application for NESCAC swimming analysis featuring:
- **Frontend**: React-based dashboard with interactive visualizations for historical data analysis and prediction interface
- **Backend**: Flask API serving machine learning models and data endpoints
- **Data Pipeline**: Automated PDF parsing, manual correction workflows, and feature engineering for NESCAC meet results
- **Models**: Both simple and advanced prediction models for finals qualification across all NESCAC events
- **Visualizations**: Comprehensive plotting system for school-specific analysis, event cutoffs, and winning times
- **Documentation**: Detailed case studies and usage examples for coaches and swimmers

## Getting Started

Each project directory contains its own documentation, data processing pipelines, and model development workflows. See the individual README files in each folder for specific setup instructions and usage guidelines.

## Technologies Used

- Python (pandas, scikit-learn, numpy)
- Machine Learning (regression, classification, ensemble methods)
- Data Visualization (matplotlib, seaborn)
- Web Scraping and Data Processing
- React.js (NESCAC frontend)
- Flask (NESCAC backend)