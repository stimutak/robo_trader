#!/usr/bin/env python3
"""
Test sentiment analysis module
"""

from robo_trader.sentiment import SimpleSentimentAnalyzer, NewsFilter

def test_sentiment_analysis():
    """Test sentiment analysis on various financial texts"""
    print("=" * 60)
    print("Testing Sentiment Analysis")
    print("=" * 60)
    
    analyzer = SimpleSentimentAnalyzer()
    
    # Test cases
    test_texts = [
        # Bullish
        ("Apple beats earnings expectations with record iPhone sales", "bullish"),
        ("Fed signals dovish stance, may pause rate hikes", "bullish"),
        ("Tesla stock surges after strong delivery numbers", "bullish"),
        
        # Bearish
        ("Company misses revenue targets, cuts guidance", "bearish"),
        ("Fed turns hawkish, signals more rate hikes ahead", "bearish"),
        ("Stock plunges after disappointing earnings report", "bearish"),
        
        # Neutral/Mixed
        ("Markets trade sideways as investors await Fed decision", "neutral"),
        ("Earnings beat expectations but guidance disappoints", "neutral"),
        
        # Negation
        ("The company did not miss earnings expectations", "bullish"),
        ("Results were not as weak as feared", "bullish"),
    ]
    
    print("\nSentiment Analysis Results:")
    print("-" * 40)
    
    for text, expected in test_texts:
        result = analyzer.analyze(text)
        status = "✅" if result.sentiment == expected else "❌"
        print(f"{status} Text: {text[:50]}...")
        print(f"   Sentiment: {result.sentiment} (expected: {expected})")
        print(f"   Score: {result.score:.2f}, Confidence: {result.confidence:.2f}")
        print(f"   Keywords: {', '.join(result.keywords[:5])}")
        print()
    
    # Test high-impact detection
    print("High-Impact Detection:")
    print("-" * 40)
    
    high_impact_texts = [
        "Fed announces surprise rate cut of 50 basis points",
        "Apple reports earnings after the bell today",
        "Minor fluctuations in pre-market trading",
        "BREAKING: Major acquisition announced",
    ]
    
    for text in high_impact_texts:
        is_high = analyzer.is_high_impact(text)
        print(f"{'🔴 HIGH' if is_high else '⚪ LOW'}: {text}")
    
    print()

def test_news_filter():
    """Test news filtering and prioritization"""
    print("Testing News Filter")
    print("-" * 40)
    
    # Create filter for specific symbols
    filter = NewsFilter(['AAPL', 'TSLA', 'SPY'])
    
    # Test relevance checking
    test_news = [
        {"title": "Apple announces new iPhone", "text": "AAPL shares rise on news"},
        {"title": "Tesla delivers record vehicles", "text": "TSLA beats expectations"},
        {"title": "Fed raises rates", "text": "Markets react to Federal Reserve decision"},
        {"title": "Random company news", "text": "XYZ corp announces earnings"},
    ]
    
    print("\nRelevance Checking:")
    for item in test_news:
        is_relevant, symbols = filter.is_relevant(item['text'], item['title'])
        if is_relevant:
            print(f"✅ Relevant to {symbols}: {item['title']}")
        else:
            print(f"❌ Not relevant: {item['title']}")
    
    # Test prioritization
    print("\nNews Prioritization:")
    prioritized = filter.prioritize_news(test_news)
    
    for i, item in enumerate(prioritized, 1):
        print(f"{i}. Priority: {item['priority']:.2f} - {item['title']}")
        print(f"   Symbols: {item['symbols']}, Sentiment: {item['sentiment'].sentiment}")

def test_performance():
    """Test performance of sentiment analysis"""
    import time
    
    print("\nPerformance Test")
    print("-" * 40)
    
    analyzer = SimpleSentimentAnalyzer()
    
    # Long text for performance testing
    long_text = """
    The Federal Reserve kept interest rates unchanged as expected but signaled a more 
    dovish stance than anticipated. Chair Powell emphasized that the central bank is 
    prepared to cut rates if economic conditions warrant, citing concerns about global 
    growth and trade tensions. Markets rallied strongly on the news, with the S&P 500 
    surging to new highs. Analysts are now pricing in multiple rate cuts for next year,
    marking a significant shift from the hawkish tone of previous meetings. This pivot
    suggests the Fed is prioritizing economic stability over inflation concerns.
    """ * 10  # Repeat 10 times for longer text
    
    start_time = time.time()
    for _ in range(100):
        result = analyzer.analyze(long_text)
    elapsed = time.time() - start_time
    
    print(f"Analyzed 100 long texts in {elapsed:.2f} seconds")
    print(f"Average time per analysis: {elapsed/100*1000:.1f} ms")
    print(f"Final sentiment: {result.sentiment} (score: {result.score:.2f})")
    
    # Compare with single short text
    short_text = "Stock beats earnings"
    start_time = time.time()
    for _ in range(1000):
        result = analyzer.analyze(short_text)
    elapsed = time.time() - start_time
    
    print(f"Analyzed 1000 short texts in {elapsed:.2f} seconds")
    print(f"Average time per analysis: {elapsed/1000*1000:.2f} ms")

if __name__ == "__main__":
    test_sentiment_analysis()
    test_news_filter()
    test_performance()
    
    print("\n" + "=" * 60)
    print("Sentiment Analysis Testing Complete!")
    print("=" * 60)
    print("\nNote: This is a lightweight rule-based analyzer.")
    print("For production, consider installing transformers for FinBERT:")
    print("  pip install transformers torch")
    print("This will provide state-of-the-art financial sentiment analysis.")