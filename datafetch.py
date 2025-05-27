import re
import requests
import time
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, ARRAY, Enum, ForeignKey, Date, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import numpy as np
from openai import OpenAI
import enum
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Database configuration with better error handling
DATABASE_URL = os.getenv("SUPABASE_POSTGRES_DB_KEY")
if not DATABASE_URL:
    raise ValueError(
        "Database URL not found! Please set the SUPABASE_POSTGRES_DB_KEY environment variable. "
        "You can do this by creating a .env file in the backend directory with: "
        "SUPABASE_POSTGRES_DB_KEY=your_database_url_here"
    )

try:
    engine = create_engine(DATABASE_URL)
    # Test the connection
    with engine.connect() as conn:
        print("Successfully connected to the database!")
except Exception as e:
    print(f"Error connecting to database: {str(e)}")
    print("Please check your database URL and make sure it's in the correct format:")
    print("postgresql://username:password@host:port/database")
    raise

Base = declarative_base()
Session = sessionmaker(bind=engine)

# Initialize OpenAI client - replace with your API key
openai_client = OpenAI(api_key="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")

# Flag for database connection
db_connected = False
conn = None

def extract_date(text):
    """Extract date from judgment text"""
    date_patterns = [
        r'DATED\s*(?::|-)?\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})',
        r'DATED\s*(?::|-)?\s*(?:THE)?\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})',
        r'Date\s*:\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})',
    ]

    text_sample = text[:1000] if text else ""

    for pattern in date_patterns:
        match = re.search(pattern, text_sample, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Clean up date string
            date_str = re.sub(r'(?:st|nd|rd|th)', '', date_str)
            date_str = re.sub(r',', '', date_str)

            try:
                # Try to parse the date
                date_obj = datetime.strptime(date_str, '%d %B %Y')
                return date_obj.date()
            except ValueError:
                # If standard format fails, try alternative formats
                try:
                    date_obj = datetime.strptime(date_str, '%d %b %Y')
                    return date_obj.date()
                except ValueError:
                    pass

    return None


def connect_to_db():
    global db_connected, conn
    try:
        conn = engine.connect()
        db_connected = True
        return True
    except Exception as e:
        print(f"Error connecting to database: {e}")
        db_connected = False
        return False

# Define court type enum
class CourtType(enum.Enum):
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    TRIBUNAL = "tribunal"

class Article2(Base):
    __tablename__ = 'article2'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    year = Column(Integer, nullable=True)
    court_type = Column(Enum(CourtType), nullable=False)
    embedding = Column(ARRAY(Float), nullable=True)

class Judgment(Base):
    __tablename__ = 'judgments'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    year = Column(Integer, nullable=True)
    court_type = Column(Enum(CourtType), nullable=False)
    citation = Column(String(255), nullable=True)  # For legal citation
    judges = Column(String(500), nullable=True)    # Names of judges
    embedding = Column(ARRAY(Float), nullable=True)
    date_decided = Column(Date, nullable=True)     # When judgment was passed

class Act(Base):
    __tablename__ = 'acts'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    year = Column(Integer, nullable=False)         # Year the act was passed
    act_number = Column(String(50), nullable=True) # Act number (e.g., "Act 21 of 2000")
    ministry = Column(String(255), nullable=True)  # Which ministry published the act
    embedding = Column(ARRAY(Float), nullable=True)
    amended = Column(Boolean, default=False)       # Whether the act has been amended

# Create the tables if they don't exist
Base.metadata.create_all(engine)

class IndianKanoonAPI:
    def __init__(self, api_token):
        self.api_token = api_token
        self.base_url = "https://api.indiankanoon.org"

    def search(self, query, page_num=0, filters=None):
        endpoint = f"{self.base_url}/search/"
        params = {"formInput": query, "pagenum": page_num}
        if filters and isinstance(filters, dict):
            params.update(filters)
        headers = {"Authorization": f"Token {self.api_token}", "Accept": "application/json"}
        try:
            response = requests.post(endpoint, headers=headers, data=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in search request: {e}")
            return None

    def get_document(self, doc_id, max_cites=5, max_cited_by=5):
        endpoint = f"{self.base_url}/doc/{doc_id}/"
        params = {"maxcites": max_cites, "maxcitedby": max_cited_by}
        headers = {"Authorization": f"Token {self.api_token}", "Accept": "application/json"}
        try:
            response = requests.post(endpoint, headers=headers, data=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in get_document request: {e}")
            return None

def extract_judges(text):
    """Extract judges names from judgment text"""
    # Look for common patterns that introduce judges
    judge_patterns = [
        r'JUDGE[S]*\s*:(.+?)(?=\n\n|\.\s|$)',
        r'HON(?:\'BLE)?\s*(?:MR\.|MS\.|MRS\.|SHRI|SMT\.|JUSTICE)(.+?)(?=\n|\.\s|and|,|\&|$)',
        r'PRESENT:(.+?)(?=\n\n|\.\s|$)',
        r'CORAM:(.+?)(?=\n\n|\.\s|$)'
    ]

    judges = []

    # Check the beginning of the text (first 1000 characters)
    text_start = text[:1000] if text else ""

    for pattern in judge_patterns:
        matches = re.finditer(pattern, text_start, re.IGNORECASE)
        for match in matches:
            judge_text = match.group(1).strip()
            # Clean up the extracted text
            judge_text = re.sub(r'\s+', ' ', judge_text)
            if judge_text and len(judge_text) < 100:  # Sanity check on length
                judges.append(judge_text)

    # Join all found judges
    if judges:
        return "; ".join(judges)

    return None

def extract_citation(text, title):
    """Extract citation from judgment text or title"""
    # Common citation patterns
    citation_patterns = [
        r'\((\d{4})\s+(?:SC|AIR|SCC|SCR|SCALE)\s+\d+(?:-\d+)?\)',  # (2022 SC 123)
        r'(\d{4})\s+(?:\(\d+\))?\s*(?:SC|AIR|SCC|SCR|SCALE)\s+\d+',  # 2022 (2) SCC 123
        r'(?:SC|AIR|SCC|SCR|SCALE)\s+(\d{4})\s+\d+',               # SCC 2022 123
    ]

    # First check the title
    for pattern in citation_patterns:
        title_match = re.search(pattern, title, re.IGNORECASE)
        if title_match:
            return title_match.group(0)

    # Then check the first part of text
    text_sample = text[:500] if text else ""
    for pattern in citation_patterns:
        text_match = re.search(pattern, text_sample, re.IGNORECASE)
        if text_match:
            return text_match.group(0)

    return None


def extract_year(text, title):
    """Extract year from text or title"""
    # Try to find year pattern in the text or title
    year_patterns = [
        r'\b(19\d{2}|20\d{2})\b',  # Match years between 1900-2099
    ]

    for pattern in year_patterns:
        # First check the title, which often contains the year
        title_match = re.search(pattern, title)
        if title_match:
            return int(title_match.group(1))

        # Then check the first 500 characters of text, where the date is often mentioned
        text_sample = text[:500] if text else ""
        text_match = re.search(pattern, text_sample)
        if text_match:
            return int(text_match.group(1))

    # Default to current year if no year found
    return 2025  # Or you could return None

def extract_act_number(text, title):
    """Extract act number from act text or title"""
    # Common act number patterns
    act_patterns = [
        r'Act\s+(?:No\.\s*)?(\d+)\s+of\s+(\d{4})',   # Act No. 21 of 2000
        r'(\d{4})\s+Act\s+(?:No\.\s*)?(\d+)',        # 2000 Act No. 21
    ]

    # First check the title
    for pattern in act_patterns:
        title_match = re.search(pattern, title, re.IGNORECASE)
        if title_match:
            if len(title_match.groups()) == 2:
                num, year = title_match.groups()
                return f"Act {num} of {year}"
            return title_match.group(0)

    # Then check the first part of text
    text_sample = text[:1000] if text else ""
    for pattern in act_patterns:
        text_match = re.search(pattern, text_sample, re.IGNORECASE)
        if text_match:
            if len(text_match.groups()) == 2:
                num, year = text_match.groups()
                return f"Act {num} of {year}"
            return text_match.group(0)

    return None

def extract_ministry(text):
    """Extract ministry from act text"""
    # Look for ministry mentions near the beginning of the act
    ministry_patterns = [
        r'MINISTRY\s+OF\s+([\w\s&,]+?)(?=\n|\.|$)',
        r'(?:ISSUED|PUBLISHED)\s+BY\s+(?:THE)?\s*MINISTRY\s+OF\s+([\w\s&,]+?)(?=\n|\.|$)'
    ]

    text_sample = text[:1500] if text else ""

    for pattern in ministry_patterns:
        match = re.search(pattern, text_sample, re.IGNORECASE)
        if match:
            ministry = match.group(1).strip()
            return ministry[:255]  # Ensure it fits in our database field

    return None

def check_if_amended(text):
    """Check if the act has been amended"""
    amendment_indicators = [
        r'(?:as\s+amended|amended\s+by)',
        r'(?:amendment)\s+act',
        r'further\s+to\s+amend'
    ]

    text_sample = text[:2000] if text else ""

    for pattern in amendment_indicators:
        if re.search(pattern, text_sample, re.IGNORECASE):
            return True

    return False

def fetch_acts_data(query, session, api_token):
    """Function to fetch acts data"""
    client = IndianKanoonAPI(api_token)
    total_saved_acts = []
    page_num = 0

    print(f"\nFetching ACTS...")

    search_filters = {
        "doctypes": "acts",
        "fromdate": "01-01-1950",
        "todate": "01-04-2025"
    }

    while True:
        search_results = client.search(query, page_num=page_num, filters=search_filters)
        if not search_results or 'docs' not in search_results or len(search_results['docs']) == 0:
            break  # No more results

        for doc in search_results.get('docs', []):
            doc_id = doc.get('tid')
            if not doc_id:
                continue

            title = doc.get('title', '')[:255]
            # Check if this act already exists in our database
            existing = session.query(Act).filter_by(name=title).first()
            if existing:
                continue

            full_doc = client.get_document(doc_id)
            if not full_doc or 'doc' not in full_doc:
                continue

            text = full_doc.get('doc', '')
            if len(text.strip()) < 50:
                continue

            name = full_doc.get('title', f'act_{doc_id}')[:255]

            # Extract metadata from the document
            year = extract_year(text, name)
            if not year:  # Acts must have a year
                continue

            act_number = extract_act_number(text, name)
            ministry = extract_ministry(text)
            amended = check_if_amended(text)

            try:
                # Create act without embedding
                new_act = Act(
                    name=name,
                    text=text,
                    year=year,
                    act_number=act_number,
                    ministry=ministry,
                    amended=amended
                )
                session.add(new_act)
                session.commit()
                total_saved_acts.append({
                    'act_id': new_act.id,
                    'name': name,
                    'year': year,
                    'act_number': act_number,
                    'amended': amended
                })
            except Exception as e:
                print(f"Error processing act {doc_id}: {e}")
                session.rollback()  # Rollback the session on error
                continue  # Skip this document and continue with the next

        print(f"ACTS - Page {page_num} done. Total so far: {len(total_saved_acts)}")

        if len(search_results.get('docs', [])) < 10:
            break  # Last page

        page_num += 1

    return total_saved_acts

def fetch_judgment_data(query, court_type, court_filter_key, session, api_token, start_page=0):
    """Function to fetch judgment data from a specific court type"""
    client = IndianKanoonAPI(api_token)
    total_saved_judgments = []
    page_num = start_page
    max_retries = 3
    retry_delay = 5  # seconds

    print(f"\nFetching JUDGMENTS from {court_type.value.upper()}...")
    print(f"Starting from page {page_num}")

    # Define filters based on court type
    search_filters = {
        "doctypes": court_filter_key,
        "fromdate": "01-01-1950",
        "todate": "01-04-2000"
    }

    while True:
        retry_count = 0
        while retry_count < max_retries:
            try:
                search_results = client.search(query, page_num=page_num, filters=search_filters)
                if not search_results or 'docs' not in search_results or len(search_results['docs']) == 0:
                    print(f"No more results found at page {page_num}")
                    return total_saved_judgments

                for doc in search_results.get('docs', []):
                    doc_id = doc.get('tid')
                    if not doc_id:
                        continue

                    title = doc.get('title', '')[:255]
                    # Check if this judgment already exists in our database
                    existing = session.query(Judgment).filter_by(name=title, court_type=court_type).first()
                    if existing:
                        continue

                    full_doc = client.get_document(doc_id)
                    if not full_doc or 'doc' not in full_doc:
                        continue

                    text = full_doc.get('doc', '')
                    if len(text.strip()) < 50:
                        continue

                    name = full_doc.get('title', f'judgment_{doc_id}')[:255]

                    # Extract metadata from the document
                    year = extract_year(text, name)
                    citation = extract_citation(text, name)
                    judges = extract_judges(text)
                    date_decided = extract_date(text)

                    try:
                        # Create judgment without embedding
                        new_judgment = Judgment(
                            name=name,
                            text=text,
                            year=year,
                            court_type=court_type,
                            citation=citation,
                            judges=judges,
                            date_decided=date_decided
                        )
                        session.add(new_judgment)
                        session.commit()
                        total_saved_judgments.append({
                            'judgment_id': new_judgment.id,
                            'name': name,
                            'year': year,
                            'citation': citation,
                            'court_type': court_type.value
                        })
                    except Exception as e:
                        print(f"Error processing judgment {doc_id}: {e}")
                        session.rollback()  # Rollback the session on error
                        continue  # Skip this document and continue with the next

                print(f"JUDGMENTS {court_type.value.upper()} - Page {page_num} done. Total so far: {len(total_saved_judgments)}")

                # Save progress to a file
                with open(f'judgment_progress_{court_type.value}.txt', 'w') as f:
                    f.write(str(page_num + 1))  # Save next page to process

                if len(search_results.get('docs', [])) < 10:
                    print(f"Last page reached at {page_num}")
                    return total_saved_judgments

                page_num += 1
                break  # Success, exit retry loop

            except Exception as e:
                retry_count += 1
                print(f"Error on page {page_num} (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached for page {page_num}. Saving progress and returning.")
                    # Save progress to a file
                    with open(f'judgment_progress_{court_type.value}.txt', 'w') as f:
                        f.write(str(page_num))  # Save current page for resume
                    return total_saved_judgments

    return total_saved_judgments

def fetch_court_data(query, court_type, court_filter_key, session, api_token):
    """Generic function to fetch data from a specific court type"""
    client = IndianKanoonAPI(api_token)
    total_saved_articles = []
    page_num = 0

    print(f"\nFetching from {court_type.value.upper()}...")

    # Define filters based on court type
    search_filters = {
        "doctypes": court_filter_key,
        "fromdate": "01-01-1950",
        "todate": "01-04-2000"
    }

    while True:
        search_results = client.search(query, page_num=page_num, filters=search_filters)
        if not search_results or 'docs' not in search_results or len(search_results['docs']) == 0:
            break  # No more results

        for doc in search_results.get('docs', []):
            doc_id = doc.get('tid')
            if not doc_id:
                continue

            title = doc.get('title', '')[:255]
            # Check if this document already exists in our database
            existing = session.query(Article2).filter_by(name=title, court_type=court_type).first()
            if existing:
                continue

            full_doc = client.get_document(doc_id)
            if not full_doc or 'doc' not in full_doc:
                continue

            text = full_doc.get('doc', '')
            if len(text.strip()) < 50:
                continue

            name = full_doc.get('title', f'document_{doc_id}')[:255]

            # Extract year from the document
            year = extract_year(text, name)

            try:
                # Create article without embedding
                new_article = Article2(
                    name=name,
                    text=text,
                    year=year,
                    court_type=court_type
                    # Removed embedding field
                )
                session.add(new_article)
                session.commit()
                total_saved_articles.append({
                    'article_id': new_article.id,
                    'name': name,
                    'year': year,
                    'court_type': court_type.value
                })
            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")
                session.rollback()  # Rollback the session on error
                continue  # Skip this document and continue with the next

        print(f"{court_type.value.upper()} - Page {page_num} done. Total so far: {len(total_saved_articles)}")

        if len(search_results.get('docs', [])) < 10:
            break  # Last page

        page_num += 1

    return total_saved_articles

@app.route('/fetch-judgments', methods=['POST'])
def fetch_judgments():
    try:
        data = request.get_json()
        query = data.get('query', 'constitution')
        api_token = "e0028ee74a551da2d5e8b3abab7a386f93bf7f2b"

        if not db_connected or conn is None:
            if not connect_to_db():
                return jsonify({
                    'status': 'Error',
                    'message': 'Database connection unavailable'
                })

        session = Session()
        results = {}

        try:
            # Check for progress files and resume from last position
            start_pages = {}
            for court_type in CourtType:
                try:
                    with open(f'judgment_progress_{court_type.value}.txt', 'r') as f:
                        start_pages[court_type] = int(f.read().strip())
                except FileNotFoundError:
                    start_pages[court_type] = 0

            # Skip Supreme Court as it's complete
            results['supreme_court'] = []
            print("Skipping Supreme Court judgments as they are complete")

            # Fetch High Court judgments starting from page 1
            high_court_results = fetch_judgment_data(
                query,
                CourtType.HIGH_COURT,
                "highcourt",
                session,
                api_token,
                start_page=0  # Start from page 1
            )
            results['high_court'] = high_court_results

            # Fetch Tribunal judgments
            tribunal_results = fetch_judgment_data(
                query,
                CourtType.TRIBUNAL,
                "tribunals",
                session,
                api_token,
                start_page=start_pages[CourtType.TRIBUNAL]
            )
            results['tribunal'] = tribunal_results

            # Count total saved judgments
            total_count = len(high_court_results) + len(tribunal_results)

            return jsonify({
                'status': 'Success',
                'message': f'Total saved judgments: {total_count}',
                'supreme_court_count': 0,  # Supreme Court is complete
                'high_court_count': len(high_court_results),
                'tribunal_count': len(tribunal_results),
                'results': results
            })

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    except Exception as e:
        return jsonify({
            'status': 'Error',
            'message': f'Error fetching judgment data: {str(e)}'
        })

@app.route('/fetch-acts', methods=['POST'])
def fetch_acts():
    try:
        data = request.get_json()
        query = data.get('query', 'constitution')
        api_token = "e0028ee74a551da2d5e8b3abab7a386f93bf7f2b"

        if not db_connected or conn is None:
            if not connect_to_db():
                return jsonify({
                    'status': 'Error',
                    'message': 'Database connection unavailable'
                })

        session = Session()

        try:
            results = fetch_acts_data(query, session, api_token)

            return jsonify({
                'status': 'Success',
                'message': f'Total saved acts: {len(results)}',
                'acts': results
            })

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    except Exception as e:
        return jsonify({
            'status': 'Error',
            'message': f'Error fetching acts data: {str(e)}'
        })

@app.route('/fetch-articles', methods=['POST'])
def fetch_articles():
    try:
        data = request.get_json()
        query = data.get('query', 'constitution')
        api_token = "e0028ee74a551da2d5e8b3abab7a386f93bf7f2b"

        if not db_connected or conn is None:
            if not connect_to_db():
                return jsonify({
                    'status': 'Error',
                    'message': 'Database connection unavailable'
                })

        session = Session()
        results = {}

        try:
            # Fetch Supreme Court articles
            # supreme_court_results = fetch_court_data(
            #     query,
            #     CourtType.SUPREME_COURT,
            #     "supremecourt",
            #     session,
            #     api_token
            # )
            # results['supreme_court'] = supreme_court_results

            # Fetch High Court articles
            # high_court_results = fetch_court_data(
            #     query,
            #     CourtType.HIGH_COURT,
            #     "highcourt",
            #     session,
            #     api_token
            # )
            # results['high_court'] = high_court_results

            # Fetch Tribunal articles
            tribunal_results = fetch_court_data(
                query,
                CourtType.TRIBUNAL,
                "tribunals",
                session,
                api_token
            )
            results['tribunal'] = tribunal_results

            # Count total saved articles
            total_count = len(tribunal_results)

            return jsonify({
                'status': 'Success',
                'message': f'Total saved articles: {total_count}',
                # 'supreme_court_count': len(supreme_court_results),
                # 'high_court_count': len(high_court_results),
                'tribunal_count': len(tribunal_results),
                'results': results
            })

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    except Exception as e:
        return jsonify({
            'status': 'Error',
            'message': f'Error fetching article data: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
