from pytube import YouTube

# Create a YouTube object
yt = YouTube('https://www.youtube.com/watch?v=a-eg3behZr0')

# Select the highest resolution stream
stream = yt.streams.get_highest_resolution()

# Download the video
stream.download()