from NoBufferVC import NoBufferVC
import pafy
import os
os.environ["PAFY_BACKEND"] = "yt-dlp"


def get_stream_capture(url):
    if "youtube" in url:
        video = pafy.new(url, basic=True)
        # [video:mp4@256x144, video:mp4@426x240, video:mp4@640x360, video:mp4@854x480, video:mp4@1280x720, video:mp4@1920x1080]
        best = video.videostreams[4]
        cap = NoBufferVC(best.url)
        return cap
    
    cap = NoBufferVC(url)
    return cap