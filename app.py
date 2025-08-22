import pandas as pd
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import nltk
import itertools
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter   # ✅ new import

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Load environment variables (not really needed now since we removed tweepy)
load_dotenv('.env')
import ssl
import requests

ssl._create_default_https_context = ssl._create_unverified_context


# --- Scraping Tweets ---
def getTweets(query, max_tweets=50):
    try:
        # Use itertools.islice to limit results quickly
        tweets = list(itertools.islice(
            sntwitter.TwitterSearchScraper(f"{query} lang:en").get_items(),
            max_tweets
        ))
        return tweets
    except Exception as e:
        print(f"❌ Error fetching tweets: {e}")
        return []


# --- Sentiment Analysis ---
def sentiment(tweets):
    positiveList = []
    negativeList = []
    neutralList = []
    for item in tweets:
        item.score = []
        sentence = item.content  # ✅ snscrape uses .content (not full_text)
        score = sid.polarity_scores(sentence)['compound']
        item.score.append(score)
        if score > 0.1:
            positiveList.append(item)
        elif score < -0.1:
            negativeList.append(item)
        else:
            neutralList.append(item)
    return positiveList, negativeList, neutralList


# --- Remove Duplicates ---
def removeDuplicate(tweets):
    uniqBox = []
    removedTweet = []
    removedList = []
    for item in tweets:
        text = item.content
        hashNumber = text.count("#")
        if text not in uniqBox:
            if 5 >= hashNumber:
                uniqBox.append(text)
                removedTweet.append(item)
        else:
            removedList.append(item)
        if 5 < hashNumber:
            removedList.append(item)
    return removedTweet, removedList


# --- Save to CSV ---
def saveToCsv(tweets):
    userNameList = []
    screenNameList = []
    tweetList = []
    NLTK = []
    for item in tweets:
        userNameList.append(item.user.displayname)
        screenNameList.append(item.user.username)
        tweetList.append(item.content)

        # Sentiment label
        score = sid.polarity_scores(item.content)['compound']
        if score > 0.1:
            NLTK.append("POSITIVE " + str(score))
        elif score < -0.1:
            NLTK.append("NEGATIVE " + str(score))
        else:
            NLTK.append("NEUTRAL " + str(score))

    df = pd.DataFrame({
        'User name': userNameList,
        'Handle': screenNameList,
        'Tweets': tweetList,
        'NLTK': NLTK
    })
    df.to_csv("./output.csv", index=False)


# --- Flask App ---
app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/searchTopic", methods=["POST", "GET"])
def searchTopic():
    output = request.form.to_dict()
    name = output.get("name", "").strip()
    if name == '':
        error = 'please put keyword'
        return render_template("index.html", error=error)

    # ✅ Limit tweets to 50 for faster results
    tweetList = getTweets(name, max_tweets=50)

    if not tweetList:  # fallback if snscrape fails
        error = f"No tweets found for '{name}' or scraping failed."
        return render_template("index.html", error=error)

    removedTweet, removedList = removeDuplicate(tweetList)

    if len(removedTweet) > 100:
        firstHundredTweets = removedTweet[0:100]
    else:
        firstHundredTweets = removedTweet

    saveToCsv(firstHundredTweets)
    positiveList, negativeList, neutralList = sentiment(firstHundredTweets)
    positiveList = sorted(positiveList, key=lambda x: x.score, reverse=True)
    neutralList = sorted(neutralList, key=lambda x: x.score)
    negativeList = sorted(negativeList, key=lambda x: x.score)

    summary = {
        'query': name,
        'total_tweets': len(tweetList),
        'removed_list': len(removedList),
        'clear_tweets': len(removedTweet),
        'sentiment_input': len(firstHundredTweets),
        'positive': len(positiveList),
        'negative': len(negativeList),
        'neutral': len(neutralList)
    }
    return render_template(
        "index.html",
        positive=positiveList,
        negative=negativeList,
        neutral=neutralList,
        removed=removedList,
        summary=summary
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
