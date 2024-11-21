from PIL import Image

def crop_image(input_path, output_path, size=(512, 512)):
    # 이미지 열기
    image = Image.open(input_path)
    
    # 이미지 크기 가져오기
    width, height = image.size
    
    # 중심을 기준으로 512x512로 크롭하기 위해 좌표 계산
    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2
    
    # 이미지 크롭
    cropped_image = image.crop((left, top, right, bottom))
    
    # 결과 이미지 저장
    cropped_image.save(output_path)

# 사용 예시
input_path = "source.png"  # 원본 이미지 파일 경로
output_path = "source_256.png"  # 출력할 이미지 파일 경로

crop_image(input_path, output_path)
