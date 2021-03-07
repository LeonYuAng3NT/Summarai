import pytube
import log_processor
import os
from moviepy import editor
from datetime import datetime
from config import media_folder, source_video_file_name, default_num_of_video_clip


class VideoAnalyzer:
    def __init__(self, url, hash, logger):
        # initialize instance variables
        self.url = str(url)
        self.hash = str(hash)
        self.logger = logger
        self.title = "youtube video"  # placeholder for the title of the youtube video

        # set up the project folder for current video
        self.project_folder = media_folder + hash + "/"
        if not os.path.exists(self.project_folder):
            os.mkdir(self.project_folder)

        # set the paths for media resources
        self.resources = {
            "source_video_path": self.project_folder + source_video_file_name + ".mp4",
            "clips_path": [],
            "audio_path": [],
            "images_path": [],
            "texts_path": []
        }

    def process_video(self):
        self._download_video()
        self._slice_video()
        self._extract_audio()
        self._extract_images()

        return self.title, self.resources

    def _download_video(self):
        # retrieve youtube video
        self.logger.info("Start processing video from url({})".format(self.url))
        youtube = pytube.YouTube(self.url)
        youtube_video = youtube.streams.get_highest_resolution()
        self.logger.info('Video "{}" retrieved'.format(youtube_video.title))

        # update video title
        self.title = youtube_video.title

        # download video and calculate the time used
        self.logger.info('Start downloading video "{}"'.format(youtube_video.title))
        download_start_time = datetime.now()

        youtube_video.download(self.project_folder, filename=source_video_file_name)
        download_finish_time = datetime.now()
        seconds_elapsed = (download_finish_time - download_start_time).total_seconds()
        video_addr = self.project_folder + source_video_file_name
        self.logger.info('Video "{}" downloaded with {} seconds:'.format(video_addr, seconds_elapsed))

    def _slice_video(self):
        # process the video into different intervals
        self.logger.info('Start analyzing video "{}" to get intervals'.format(self.title))
        slice_interval = None

        # retrieve the source clip
        source_clip = editor.VideoFileClip(self.resources["source_video_path"])

        # slice the video into same duration clips based on default intervals
        self.logger.info('Intervals for video "{}" is mising, get default intervals'.format(self.title))
        if slice_interval == None:
            duration = source_clip.duration
            clip_length = duration / default_num_of_video_clip
            clip_time_list = []
            for i in range(0, default_num_of_video_clip + 1):
                clip_time_list.append(clip_length * i)
            self.logger.info("Default intervals for video {} generated: [{}] ".format(self.title, clip_time_list))

        # slice the video into sub clips according to the clip_time_addr
        total_slicing_time = 0
        self.logger.info('Start slicing video "{}"'.format(self.title))
        for i in range(0, len(clip_time_list) - 1):
            # start counting processing time for the current clip
            clip_start_time = datetime.now()

            # determine the starting and ending time of the clip
            current_clip_start_time = clip_time_list[i]
            current_clip_end_time = clip_time_list[i + 1]

            # slice the source video and save the current sub clip
            subclip = source_clip.subclip(current_clip_start_time, current_clip_end_time)
            current_clip_addr = self.project_folder + "clip{}.mp4".format(i + 1)
            self.resources["clips_path"].append(current_clip_addr)
            subclip.write_videofile(current_clip_addr, logger=None)

            clip_finish_time = datetime.now()
            seconds_elapsed = (clip_finish_time - clip_start_time).total_seconds()
            total_slicing_time += seconds_elapsed
            self.logger.info('Video "{}", Clip #{} ({},{}) is sliced and saved with {} seconds'
                             .format(self.title, i + 1, current_clip_start_time,
                                     current_clip_end_time, seconds_elapsed))

        self.logger.info('Video "{}" has been sliced into {} clips with {} seconds'
                         .format(self.title, len(self.resources["clips_path"]), total_slicing_time))

    def _extract_audio(self):
        self.logger.info('Start extracting audio for "{}"'.format(self.title))
        total_extracting_time = 0

        # process each clip for the video
        counter = 0
        for each_clip_path in self.resources["clips_path"]:
            # start counting processing time for the current clip
            clip_start_time = datetime.now()
            counter += 1

            # extract audio and save it in the corresponding path
            clip = editor.VideoFileClip(each_clip_path)
            audio = clip.audio
            audio_addr = each_clip_path.replace("mp4", "mp3")
            self.resources["audio_path"].append(audio_addr)
            audio.write_audiofile(audio_addr, logger=None)

            # count time used and report
            clip_finish_time = datetime.now()
            seconds_elapsed = (clip_finish_time - clip_start_time).total_seconds()
            total_extracting_time += seconds_elapsed
            self.logger.info('Video "{}", audio for Clip #{} is sliced and saved with {} seconds'
                             .format(self.title, counter, seconds_elapsed))

        self.logger.info('{} audios for "{}" have been extracted with  {} seconds'
                         .format(counter, self.title, total_extracting_time))

    def _extract_images(self):
        self.logger.info('Start extracting images for "{}"'.format(self.title))
        total_extracting_time = 0

        # process each clip for the video
        counter = 0
        for each_clip_path in self.resources["clips_path"]:
            # start counting processing time for the current clip
            clip_start_time = datetime.now()
            counter += 1

            # extract image and save it in the corresponding path
            clip = editor.VideoFileClip(each_clip_path)
            image_addr = each_clip_path.replace("mp4", "png")
            self.resources["images_path"].append(image_addr)
            clip.save_frame(image_addr, (clip.duration - 0.01))

            # count time used and report
            clip_finish_time = datetime.now()
            seconds_elapsed = (clip_finish_time - clip_start_time).total_seconds()
            total_extracting_time += seconds_elapsed
            self.logger.info('Video "{}", images for Clip #{} is sliced and saved with {} seconds'
                             .format(self.title, counter, seconds_elapsed))

        self.logger.info('{} images for "{}" have been extracted with  {} seconds'
                         .format(counter, self.title, total_extracting_time))


# def extract_frame(video_addr, frame_time=0.1):
#     clip = editor.VideoFileClip(video_addr)
#     img_addr = video_addr.replace("mp4","png")
#     clip.save_frame(img_addr,frame_time)
#     return img_addr

# def batch_extract_cover_img(video_addr_list):
#     img_addr_list = []
#     for each in video_addr_list:
#         img_addr = extract_frame(each)
#         img_addr_list.append(img_addr)
#     return  img_addr_list


def clear_media():
    media_files = []

    for filename in os.listdir(media_folder):
        file_path = os.path.join(media_folder, filename)
        try:
            if os.path.isfile(file_path):
                if (file_path.endswith(".mp4")
                        or file_path.endswith(".mp3")
                        or file_path.endswith(".png")):
                    os.remove(file_path)
                    media_files.append(file_path)
        except Exception as e:
            logger.warn('Failed to delete %s. Reason: %s' % (file_path, e))

    msg = "The following media have been cleared:\n"
    for each in media_files:
        msg += "  " + each + "\n"

    logger.info(msg)


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=ww95zvvhX0c&list=PLfetWr5y3KmmAPUcfMnepril3nHDNTJ0a"
    hash = "hashcode_test"
    logger = log_processor.get_logger("hashcode_test")
    media_processor = MediaProcessor(url, hash, logger)

    media_processor.download_video()
    media_processor.slice_video()
    media_processor.extract_audio()
    media_processor.extract_images()
    # test_url = "https://www.youtube.com/watch?v=IwFpiywyjcE"

    # clear_media()
    # video_addr = download_from_youtube(short_video_url)
    # print("Video addr = {}".format(video_addr))
    # clip_addr = slice_video(video_addr)
    # print("clip addr = {}".format(clip_addr))
    #
    #
    # img_addr_list = batch_extract_cover_img(clip_addr)
    # print("img_addr = {}".format(img_addr_list))
    #
    # audio_addr_list = batch_extract_audio(clip_addr)
    # print("audio_addr_list = {}".format(audio_addr_list))

