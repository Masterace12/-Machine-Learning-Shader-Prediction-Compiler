#!/usr/bin/env python3
"""
Creates a simple icon for the Shader Predictive Compiler
"""

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("PIL not available, creating empty icon file")
    with open("icon.png", "wb") as f:
        # Write minimal PNG header for 1x1 transparent pixel
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\xdac\xf8\x0f\x00\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
    print("Created placeholder icon.png")
    exit(0)

# Create a 256x256 icon
size = 256
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background gradient
for i in range(size):
    color = int(50 + (i / size) * 50)
    draw.rectangle([(0, i), (size, i+1)], fill=(color, color, color+20, 255))

# Draw gear/cog shape (representing compilation/processing)
center = size // 2
radius = size // 3

# Outer circle
draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
             fill=(100, 150, 255, 255), outline=(80, 120, 200, 255), width=4)

# Inner circle
inner_radius = radius // 2
draw.ellipse([center-inner_radius, center-inner_radius, 
              center+inner_radius, center+inner_radius], 
             fill=(60, 90, 150, 255))

# Draw gear teeth
import math
teeth = 8
for i in range(teeth):
    angle = (2 * math.pi * i) / teeth
    x1 = center + int(radius * 0.8 * math.cos(angle))
    y1 = center + int(radius * 0.8 * math.sin(angle))
    x2 = center + int(radius * 1.1 * math.cos(angle))
    y2 = center + int(radius * 1.1 * math.sin(angle))
    
    # Draw tooth
    tooth_width = radius // 5
    draw.ellipse([x2-tooth_width//2, y2-tooth_width//2, 
                  x2+tooth_width//2, y2+tooth_width//2], 
                 fill=(100, 150, 255, 255))

# Add text
try:
    # Try to use a monospace font
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
except:
    font = ImageFont.load_default()

# Draw "SPC" text in center
text = "SPC"
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
draw.text((center - text_width//2, center - text_height//2), text, 
          fill=(255, 255, 255, 255), font=font)

# Save icon
img.save("icon.png", "PNG")
print("Created icon.png successfully!")

# Also create smaller versions
for new_size in [128, 64, 48, 32, 16]:
    resized = img.resize((new_size, new_size), Image.Resampling.LANCZOS)
    resized.save(f"icon_{new_size}.png", "PNG")
    print(f"Created icon_{new_size}.png")