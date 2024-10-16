from datetime import datetime
import time
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import feedparser
from dateutil import parser
from bs4 import BeautifulSoup
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql+pymysql://root:12345@localhost/employee')  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the fine-tuned model and tokenizer
path_to_model = "fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(path_to_model).to(device)
tokenizer = AutoTokenizer.from_pretrained(path_to_model)

category_to_number = {
    'WELLNESS': 0,
    'POLITICS': 1,
    'ENTERTAINMENT': 2,
    'TRAVEL': 3,
    'STYLE & BEAUTY': 4,
    'PARENTING': 5,
    'FOOD & DRINK': 6,
    'WORLD NEWS': 7,
    'BUSINESS': 8,
    'SPORTS': 9
}

class RSSFeed(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(255), nullable=False, unique=True)
    summary = db.Column(db.Text, nullable=False)
    published = db.Column(db.DateTime, nullable=True, default=None) 
    cat = db.Column(db.String(30), nullable=False)

    def __repr__(self):
        return f'<RSSFeed {self.title}>'

with app.app_context():
    db.create_all()

rss_url_list = [
  #'https://rss.app/feeds/THdjvV6HaRFAOHK9.xml',
 # 'https://rss.app/feeds/dmsa8GEmKxTPJWnW.xml',
 # 'https://rss.app/feeds/IuAqrT63zfZyRdJX.xml',
  #'https://rss.app/feeds/PSnt6T3eGnIVdtgj.xml',
  #'http://rss.cnn.com/rss/cnn_topstories.rss',
 # 'https://moxie.foxnews.com/google-publisher/latest.xml',
 # 'https://abcnews.go.com/abcnews/entertainmentheadlines',
 # 'https://moxie.foxnews.com/google-publisher/health.xml',
 # 'https://moxie.foxnews.com/google-publisher/sports.xml',
 # 'https://rss.app/feeds/XoIkVuVS0cFoBfWh.xml',
 # 'https://rss.app/feeds/Faj9gGzeBVGt275A.xml',
 # 'https://rss.app/feeds/vcLJUewPUvpQIytP.xml',
  #'https://rss.app/feeds/CZAfM8xUJK4sYfRb.xml',
  #'https://rss.app/feeds/KelSj6mVpFW2op9A.xml',
 # 'https://rss.app/feeds/04SP7wIeLbLvCu2a.xml',

]

def analysis(clean_summary):
    return BeautifulSoup(clean_summary, "html.parser").get_text() if is_valid_html(clean_summary) else clean_summary

def is_valid_html(content):
    return bool(re.search(r"<[a-zA-Z]+[^>]*>", content))

def parse_published_date(date_string):
    try:
        return datetime.strptime(date_string, '%a, %d %b %Y %H:%M:%S %z')
    except ValueError:
        return parser.parse(date_string)

def log_entry_details(title, link, summary, published):
    logging.info(f"Title: {title}")
    logging.info(f"Link: {link}")
    logging.info(f"Summary: {summary}")
    logging.info(f"Date: {published if published else 'No Date'}")

def fetch_and_store_rss():
    c = 0
    for rss_url in rss_url_list:
        feed = feedparser.parse(rss_url)
        new_feeds = []

        for entry in feed.entries:
            title = getattr(entry, 'title', None)
            summary = getattr(entry, 'summary', None)
            link = getattr(entry, 'link', None)

            try:
                clean_summary = analysis(summary)
            except Exception as e:
                logging.error(f"Error processing summary: {e}")
                clean_summary = 'Error processing summary'
            
            published = parse_published_date(entry.published) if 'published' in entry else None
                
            log_entry_details(title, link, clean_summary, published)

            if title and clean_summary:
                test_sample = {
                    'headline': title,
                    'short_description': clean_summary if clean_summary else ''
                } 

                tokenized_test_sample = tokenizer(
                    test_sample['headline'] + " " + test_sample['short_description'],
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                tokenized_test_sample = {key: value.to(device) for key, value in tokenized_test_sample.items()}

                try:
                    with torch.no_grad():
                        outputs = model(**tokenized_test_sample)

                    predicted_label = torch.argmax(outputs.logits, dim=1).item()
                    predicted_category = {v: k for k, v in category_to_number.items()}.get(predicted_label, 'Unknown')

                    logging.info(f"Predicted category: {predicted_category}")
                    logging.info("================")
                except Exception as e:
                    logging.error(f"Error during prediction: {e}")
                    predicted_category = 'Unknown'

            if db.session.query(RSSFeed).filter_by(link=link.rstrip('/')).first() is None:
                new_feed = RSSFeed(
                    title=title,
                    link=link,
                    summary=clean_summary,
                    published=published if published else None,
                    cat=predicted_category
                )
                new_feeds.append(new_feed)
            else:
                logging.info(f"Duplicate entry found for link: {link}")

            time.sleep(0.15)

        if new_feeds:
            db.session.bulk_save_objects(new_feeds)
            db.session.commit()
            c += len(new_feeds)
            logging.info(f"Stored {len(new_feeds)} new entries. Total is {c}")
        else:
            logging.info("No new entries to store.")

# Run the function to fetch and store RSS feeds
with app.app_context():
    fetch_and_store_rss()
