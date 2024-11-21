from moviepy.editor import VideoFileClip

def resize_video(input_path, output_path, size=(512, 512)):
    clip = VideoFileClip(input_path)
    
    resized_clip = clip.resize(newsize=size)
    
    resized_clip.write_videofile(output_path, codec='libx264')

input_path = "video3.mp4"  # 원본 동영상 파일 경로
output_path = "video3_256.mp4"  # 출력할 동영상 파일 경로

resize_video(input_path, output_path)
