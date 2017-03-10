
import tweepy


consumer_key = ''
consumer_secret = ''


def fetch_tweets(handle):

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth)

    tweets = []

    new_tweets = api.user_timeline(screen_name=handle, count=200)

    while len(new_tweets) > 0:
        tweets.extend(new_tweets)

        oldest = tweets[-1].id - 1

        ## use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(
            screen_name=handle, count=200, max_id=oldest)

    with open('%s_tweets.txt' % handle, 'w') as f:
        for tweet in tweets:
            f.write('%s\n' % tweet.text)


if __name__ == '__main__':
    fetch_tweets('realdonaldtrump')
