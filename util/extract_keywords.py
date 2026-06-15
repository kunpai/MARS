import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def extract_keywords(paragraph, num_keywords=5):
    """
    Extracts a few keywords from a given paragraph.
    
    Args:
        paragraph (str): The input text.
        num_keywords (int): Number of keywords to extract.
        
    Returns:
        list: A list of extracted keywords.
    """
    # Tokenize the paragraph into words
    words = word_tokenize(paragraph.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_words = [
        word for word in words if word not in stop_words and word not in string.punctuation
    ]
    
    # Count the frequency of each word
    word_freq = Counter(filtered_words)
    
    # Extract the most common keywords
    keywords = [word for word, freq in word_freq.most_common(num_keywords)]
    
    return keywords
