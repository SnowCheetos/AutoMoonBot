from PIL import Image

def create_gif(frame_folder, output_file, duration=500, max_len=1000):
    frames = []
    for i in range(1, max_len):
        frames.append(Image.open(f"{frame_folder}/{i}.png").convert("RGB"))
    
    if not frames:
        raise ValueError("No frames found in the specified folder.")
    
    frames[0].save(output_file, format='GIF', append_images=frames[1:max_len], save_all=True, duration=duration, loop=0)

if __name__ == "__main__":
    frame_folder = 'logs/frames'
    output_file  = 'media/demo.gif'
    create_gif(frame_folder, output_file, 100, 700)
