from youtube_transcript_api import YouTubeTranscriptApi
from google.cloud import language_v1
import argparse
import io
import json
import os
import numpy
import six

key_subject_list = []
key_terminology_list = []

def get_text_from_video(video_id):
    
    result = YouTubeTranscriptApi.get_transcript(video_id,languages=['de','en'])
     
    # Remove Ads
    for i in range(0,5):
        result.remove(result[i])
    return result


def classify_content(text, verbose=True):
    """Classify the input text into categories. """

    language_client = language_v1.LanguageServiceClient()

    document = language_v1.Document(
        content=text, type_=language_v1.Document.Type.PLAIN_TEXT
    )
    response = language_client.classify_text(request={'document': document})
    categories = response.categories

    result = {}

    for category in categories:
        result[category.name] = category.confidence

    if verbose:
        print(text)
        for category in categories:
            print(u"=" * 20)
            print(u"{:<16}: {}".format("category", category.name))
            print(u"{:<16}: {}".format("confidence", category.confidence))

    return result

def divide_steps(transcript_list, time_frame_list):
    index = 0
    curr = 0
    result = []
    temp_text = ""

    print(time_frame_list)
    while index < (len(time_frame_list)-1) :
        if transcript_list[curr]["start"] < time_frame_list[index]:
            temp_text += " " + transcript_list[curr]["text"]
            curr+=1;
            
        elif transcript_list[curr]["start"] >= time_frame_list[index]:
            result.append(temp_text)
            index+=1
            curr+=1;
            temp_text = ""
    temp_text = ""
    for i in range(curr,len(transcript_list)):
        temp_text += transcript_list[i]["text"]
    result.append(temp_text)
    return result

def analyze_entity_sentiment(text_content):
    """
    Analyzing Entity Sentiment in a String

    Args:
      text_content The text content to analyze
    """

    words = []
    client = language_v1.LanguageServiceClient()

    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entity_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Loop through entitites returned from the API
    for entity in response.entities:
        print(u"Representative name for the entity: {}".format(entity.name))
        # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
        curr_type = language_v1.Entity.Type(entity.type_).name
        print(u"Entity type: {}".format(curr_type))
        print(u"Salience score: {}".format(entity.salience))
        # Get the aggregate sentiment expressed for this entity in the provided document.
        sentiment = entity.sentiment
        print(u"Entity sentiment score: {}".format(sentiment.score))
        print(u"Entity sentiment magnitude: {}".format(sentiment.magnitude))
        if(sentiment.score == 0.0 and sentiment.magnitude == 0.0 and entity.salience >= 0.0001 and curr_type == "OTHER"):
            words.append(entity.name)
        if ((sentiment.magnitude + sentiment.score == 0.0 or sentiment.score - sentiment.magnitude == 0.0) and (curr_type == "COMMON" or curr_type == "OTHER") and entity.salience >= 0.001):
            words.append(entity.name)
        for metadata_name, metadata_value in entity.metadata.items():
            print(u"{} = {}".format(metadata_name, metadata_value))

        # Loop over the mentions of this entity in the input document.
        # The API currently supports proper noun mentions.
        for mention in entity.mentions:
            print(u"Mention text: {}".format(mention.text.content))
            # Get the mention type, e.g. PROPER for proper noun
            print(
                u"Mention type: {}".format(language_v1.EntityMention.Type(mention.type_).name)
            )
    return words
    print(u"Language of the text: {}".format(response.language))

if __name__ == "__main__":
    video_id = ""

    transcript_list = get_text_from_video(video_id)
    