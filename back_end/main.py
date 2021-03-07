from typing import Optional
from fastapi import FastAPI, BackgroundTasks
import time
import log_processor
import re
import requests
import hashlib
from datetime import datetime
from config import time_format, server_responses, \
    server_log_file_name, url_hashing_character_count,\


app = FastAPI()

# set up logger
current_time = datetime.now()
current_session_time = current_time.strftime(time_format)
logger = log_processor.get_logger(current_session_time,server_log_file_name)
logger.info("FastAPI server started")

@app.get("/")
async def process_video_async_api(background_tasks:BackgroundTasks):
    background_tasks.add_task(process_video)
    return{"Summarai":"the best summary to help you pass your exam",
           'statis':201}


def process_video(url,email):

    # validate parameters
    valid_url = check_url(url)
    valid_email = check_email(email)
    if not valid_url:
        return server_responses["failed_post_url"]
    if not valid_email:
        return server_responses["failed_post_email"]
    logger.info("new request: url is {} and email is {}".format(url, email))

    # make an async call to process the video and proceed to return response
    extract_note.delay(url, email)
    return server_responses["successful_post"]



api.add_resource(GetYouTubeTask,'/')


# Helper methods
def get_hash(s):
    hash_object = hashlib.sha512(s.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex[0:url_hashing_character_count]

def check_url(url):
    if not (url.startswith("https://www.youtube.com/watch?v=")
        or url.startswith("www.youtube.com/watch?v=")):
        return False

    try:
        request = requests.get(url)
        if request.status_code != 200:
            print("3")
            return False
    except:
        print("Exception!")
        return False

    return True

def check_email(email):
    regex_for_email = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    return re.search(regex_for_email, email)
