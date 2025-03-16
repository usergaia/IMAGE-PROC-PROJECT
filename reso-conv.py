from PIL import Image

def resize_image_pil(input_path, output_path, width, height):
    img = Image.open(input_path)
    img = img.resize((width, height), Image.LANCZOS)
    img.save(output_path)
    print(f"Resized image saved as {output_path}")
    

# Example usage
resize_image_pil("input.jpg", "foldername/outputimage.jpg", width=256, height=256)

