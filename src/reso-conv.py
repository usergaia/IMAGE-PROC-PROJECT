from PIL import Image

def resize_image_pil(input_path, output_path, width, height):
    img = Image.open(input_path)
    img = img.resize((width, height), Image.LANCZOS)
    img.save(output_path)
    print(f"Resized image saved as {output_path}")
    

# Example usage
resize_image_pil("images/mouse.jpg", "images/mouse/mouse_256.jpg", width=256, height=256)
resize_image_pil("images/mouse.jpg", "images/mouse/mouse_299.jpg", width=299, height=299)
resize_image_pil("images/mouse.jpg", "images/mouse/mouse_512.jpg", width=512, height=512)
resize_image_pil("images/mouse.jpg", "images/mouse/mouse_1024.jpg", width=1024, height=1024)


