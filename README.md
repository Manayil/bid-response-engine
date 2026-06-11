# Bid and Proposal Response Engine

## Project Overview
This application is designed to streamline the procurement process by automating the analysis of Request for Proposal (RFP) documents. The engine utilizes large language models and historical data to help bid managers make informed Go/No-Go decisions and generate initial technical responses.

## System Modules
- **Document Ingestion**: Extracts text from PDF documents to identify technical requirements.
- **Historical Analysis**: References a dataset of 120 past bids to calculate win probabilities based on specific sectors.
- **Capability Mapping**: Cross-references requirements against a library of 50 past projects and certifications to find the best organizational fit.
- **Automated Drafting**: Generates executive summaries and compliance checklists based on extracted document context.

## Technical Architecture
- **Backend**: Python 3.10
- **Artificial Intelligence**: Llama-3.3-70b via the Groq API
- **Data Management**: Pandas for structured CSV analysis
- **Document Processing**: PyMuPDF for high-fidelity text extraction
- **Interface**: Streamlit Web Framework

## Installation and Setup
1. Clone the repository to your local machine.
2. Install the required dependencies using:
   pip install -r requirements.txt
3. Create a .env file in the root directory and add your API key:
   GROQ_API_KEY=your_api_key_here
4. Run the application:
   streamlit run app.py

## Project Structure
- app.py: Core application logic and user interface.
- data/: Directory containing bid_history.csv and capability_library.csv.
- .gitignore: Configuration to exclude environment variables and temporary files.
