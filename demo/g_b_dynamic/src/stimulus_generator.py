import numpy as np
import cv2 # Using OpenCV for drawing shapes, can be replaced if not desired

def generate_simple_shape(image_size=(32, 32), shape_type='circle', noise_level=0.1):
    """
    Generates a 2D image with a simple geometric shape (circle, square, triangle)
    and adds some Gaussian noise.

    Args:
        image_size (tuple): (height, width) of the image.
        shape_type (str): Type of shape: 'circle', 'square', 'triangle'.
        noise_level (float): Standard deviation of Gaussian noise to add.

    Returns:
        numpy.ndarray: Grayscale image with the shape, values in [0, 1].
    """
    image = np.zeros(image_size, dtype=np.float32)
    height, width = image_size
    color = 255 # White shape on black background before normalization

    # Shape parameters (ensure shape is not too small or too large)
    min_dim = min(height, width)
    center_x, center_y = width // 2, height // 2
    size = min_dim // 3 # Base size for shapes

    if shape_type == 'circle':
        radius = size
        cv2.circle(image, (center_x, center_y), radius, color, -1) # -1 for filled
    elif shape_type == 'square':
        side_length = int(size * 1.5) # Adjust for visual balance
        top_left_x = center_x - side_length // 2
        top_left_y = center_y - side_length // 2
        cv2.rectangle(image, (top_left_x, top_left_y), 
                        (top_left_x + side_length, top_left_y + side_length), color, -1)
    elif shape_type == 'triangle': # Equilateral triangle
        side = int(size * 2)
        h = int(side * np.sqrt(3) / 2) # Height of equilateral triangle
        pt1 = (center_x, center_y - h // 2)
        pt2 = (center_x - side // 2, center_y + h // 2)
        pt3 = (center_x + side // 2, center_y + h // 2)
        triangle_cnt = np.array([pt1, pt2, pt3], dtype=np.int32)
        cv2.drawContours(image, [triangle_cnt], 0, color, -1)
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}. Supported: circle, square, triangle.")

    # Normalize to [0, 1]
    if np.max(image) > 0:
        image = image / np.max(image)

    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)
        
    return image

def generate_noise_field(image_size=(32, 32), noise_intensity=0.5):
    """
    Generates a 2D noise field.

    Args:
        image_size (tuple): (height, width) of the image.
        noise_intensity (float): Max value of the uniform noise, scaled to [0,1].

    Returns:
        numpy.ndarray: Grayscale noise image, values in [0, 1].
    """
    image = np.random.uniform(0, noise_intensity, image_size).astype(np.float32)
    return np.clip(image, 0, 1)

def generate_line(image_size=(32, 32), orientation='horizontal', noise_level=0.1):
    """
    Generates a 2D image with a single line (horizontal or vertical)
    and adds some Gaussian noise.

    Args:
        image_size (tuple): (height, width) of the image.
        orientation (str): 'horizontal' or 'vertical'.
        noise_level (float): Standard deviation of Gaussian noise to add.

    Returns:
        numpy.ndarray: Grayscale image with the line, values in [0, 1].
    """
    image = np.zeros(image_size, dtype=np.float32)
    height, width = image_size
    color = 255 # White line on black background before normalization
    thickness = max(1, min(height, width) // 16) # Line thickness relative to image size

    if orientation == 'horizontal':
        y_center = height // 2
        cv2.line(image, (0, y_center), (width - 1, y_center), color, thickness)
    elif orientation == 'vertical':
        x_center = width // 2
        cv2.line(image, (x_center, 0), (x_center, height - 1), color, thickness)
    else:
        raise ValueError(f"Unknown orientation: {orientation}. Supported: horizontal, vertical.")

    if np.max(image) > 0:
        image = image / np.max(image)

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)
        
    return image

def generate_squircle(image_size=(32,32), noise_level=0.1):
    """
    Generates a 2D image with a squircle.
    A squircle is a shape intermediate between a square and a circle.
    This implementation draws a rounded rectangle with high corner radius.
    """
    image = np.zeros(image_size, dtype=np.float32)
    height, width = image_size
    color = 255
    min_dim = min(height, width)
    center_x, center_y = width // 2, height // 2
    
    rect_width = int(min_dim * 0.6)
    rect_height = int(min_dim * 0.6)
    top_left_x = center_x - rect_width // 2
    top_left_y = center_y - rect_height // 2
    
    # High corner radius to make it look like a squircle
    corner_radius = min(rect_width, rect_height) // 3 

    # OpenCV doesn't have a direct rounded rectangle fill. We can draw lines and circles.
    # For simplicity, let's use a rectangle and then blur it significantly, 
    # or use a simpler approximation. A true squircle is more complex.
    # As a proxy: draw a circle and a square and blend them, or just a highly rounded rect.
    # For now, let's use cv2.rectangle with a large thickness and then fill, or draw it manually.
    # A simpler approach for a filled rounded rectangle:
    cv2.rectangle(image, 
                  (top_left_x + corner_radius, top_left_y),
                  (top_left_x + rect_width - corner_radius, top_left_y + rect_height),
                  color, -1)
    cv2.rectangle(image, 
                  (top_left_x, top_left_y + corner_radius),
                  (top_left_x + rect_width, top_left_y + rect_height - corner_radius),
                  color, -1)
    cv2.circle(image, (top_left_x + corner_radius, top_left_y + corner_radius), corner_radius, color, -1)
    cv2.circle(image, (top_left_x + rect_width - corner_radius, top_left_y + corner_radius), corner_radius, color, -1)
    cv2.circle(image, (top_left_x + corner_radius, top_left_y + rect_height - corner_radius), corner_radius, color, -1)
    cv2.circle(image, (top_left_x + rect_width - corner_radius, top_left_y + rect_height - corner_radius), corner_radius, color, -1)

    if np.max(image) > 0:
        image = image / np.max(image)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)
    return image

def generate_star(image_size=(32,32), num_points=5, noise_level=0.1):
    """
    Generates a 2D image with a star shape.
    """
    image = np.zeros(image_size, dtype=np.float32)
    height, width = image_size
    color = 255
    center_x, center_y = width // 2, height // 2
    
    outer_radius = min(height, width) // 3
    inner_radius = outer_radius // 2
    
    points = []
    for i in range(num_points * 2):
        angle = i * np.pi / num_points - np.pi / 2 # Start from top
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        points.append((x,y))
    
    star_cnt = np.array(points, dtype=np.int32)
    cv2.drawContours(image, [star_cnt], 0, color, -1)

    if np.max(image) > 0:
        image = image / np.max(image)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)
    return image

def generate_hexagon(image_size=(32,32), noise_level=0.1):
    """
    Generates a 2D image with a hexagon shape.
    """
    image = np.zeros(image_size, dtype=np.float32)
    height, width = image_size
    color = 255
    center_x, center_y = width // 2, height // 2
    radius = min(height, width) // 3
    
    points = []
    for i in range(6):
        angle = i * np.pi / 3 # 60 degrees per point
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        points.append((x,y))
        
    hexagon_cnt = np.array(points, dtype=np.int32)
    cv2.drawContours(image, [hexagon_cnt], 0, color, -1)

    if np.max(image) > 0:
        image = image / np.max(image)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)
    return image

def generate_threat_shape(image_size=(32,32), noise_level=0.1):
    """
    Generates a "threat" shape - e.g., a jagged, irregular polygon.
    """
    image = np.zeros(image_size, dtype=np.float32)
    height, width = image_size
    color = 255
    center_x, center_y = width // 2, height // 2
    
    # Create a somewhat random, jagged shape
    num_vertices = np.random.randint(7, 12) # More vertices for jaggedness
    points = []
    base_radius = min(height, width) // 3
    for i in range(num_vertices):
        angle = i * 2 * np.pi / num_vertices
        # Vary radius to make it irregular and jagged
        radius_variation = base_radius * (0.7 + np.random.rand() * 0.6) 
        x = int(center_x + radius_variation * np.cos(angle))
        y = int(center_y + radius_variation * np.sin(angle))
        points.append((x,y))
        
    threat_cnt = np.array(points, dtype=np.int32)
    cv2.drawContours(image, [threat_cnt], 0, color, -1)

    if np.max(image) > 0:
        image = image / np.max(image)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)
    return image

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def display_image(img_array, title=''):
        plt.imshow(img_array, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.axis('off')
        plt.show()

    print("Generating Phase 1 Stimuli Examples...")

    # Test shapes
    circle_img = generate_simple_shape(shape_type='circle', noise_level=0.05)
    # display_image(circle_img, 'Circle with Noise')

    square_img = generate_simple_shape(shape_type='square', noise_level=0.1)
    # display_image(square_img, 'Square with Noise')

    triangle_img = generate_simple_shape(shape_type='triangle', noise_level=0.15)
    # display_image(triangle_img, 'Triangle with Noise')

    # Test noise field
    noise_img = generate_noise_field(noise_intensity=0.7)
    # display_image(noise_img, 'Noise Field')

    # Test lines
    h_line_img = generate_line(orientation='horizontal', noise_level=0.05)
    # display_image(h_line_img, 'Horizontal Line with Noise')

    v_line_img = generate_line(orientation='vertical', noise_level=0.05)
    # display_image(v_line_img, 'Vertical Line with Noise')

    print(f"Circle image shape: {circle_img.shape}, dtype: {circle_img.dtype}, min: {np.min(circle_img):.2f}, max: {np.max(circle_img):.2f}")
    print(f"Square image shape: {square_img.shape}, dtype: {square_img.dtype}, min: {np.min(square_img):.2f}, max: {np.max(square_img):.2f}")
    print(f"Triangle image shape: {triangle_img.shape}, dtype: {triangle_img.dtype}, min: {np.min(triangle_img):.2f}, max: {np.max(triangle_img):.2f}")
    print(f"Noise field shape: {noise_img.shape}, dtype: {noise_img.dtype}, min: {np.min(noise_img):.2f}, max: {np.max(noise_img):.2f}")
    print(f"Horizontal line shape: {h_line_img.shape}, dtype: {h_line_img.dtype}, min: {np.min(h_line_img):.2f}, max: {np.max(h_line_img):.2f}")
    print(f"Vertical line shape: {v_line_img.shape}, dtype: {v_line_img.dtype}, min: {np.min(v_line_img):.2f}, max: {np.max(v_line_img):.2f}")

    # Example with different size
    large_circle = generate_simple_shape(image_size=(64,64), shape_type='circle', noise_level=0.0)
    # display_image(large_circle, 'Large Circle No Noise')
    print(f"Large circle shape: {large_circle.shape}")

    print("\nGenerating Phase 2 Stimuli Examples...")
    squircle_img = generate_squircle(noise_level=0.05)
    # display_image(squircle_img, 'Squircle')

    star_img = generate_star(num_points=5, noise_level=0.05)
    # display_image(star_img, '5-Point Star')

    hexagon_img = generate_hexagon(noise_level=0.05)
    # display_image(hexagon_img, 'Hexagon')

    threat_img = generate_threat_shape(noise_level=0.05)
    # display_image(threat_img, 'Threat Shape')

    print(f"Circle image shape: {circle_img.shape}, dtype: {circle_img.dtype}, min: {np.min(circle_img):.2f}, max: {np.max(circle_img):.2f}")
    print(f"Square image shape: {square_img.shape}, dtype: {square_img.dtype}, min: {np.min(square_img):.2f}, max: {np.max(square_img):.2f}")
    print(f"Triangle image shape: {triangle_img.shape}, dtype: {triangle_img.dtype}, min: {np.min(triangle_img):.2f}, max: {np.max(triangle_img):.2f}")
    print(f"Noise field shape: {noise_img.shape}, dtype: {noise_img.dtype}, min: {np.min(noise_img):.2f}, max: {np.max(noise_img):.2f}")
    print(f"Horizontal line shape: {h_line_img.shape}, dtype: {h_line_img.dtype}, min: {np.min(h_line_img):.2f}, max: {np.max(h_line_img):.2f}")
    print(f"Vertical line shape: {v_line_img.shape}, dtype: {v_line_img.dtype}, min: {np.min(v_line_img):.2f}, max: {np.max(v_line_img):.2f}")
    print(f"Squircle shape: {squircle_img.shape}, dtype: {squircle_img.dtype}, min: {np.min(squircle_img):.2f}, max: {np.max(squircle_img):.2f}")
    print(f"Star shape: {star_img.shape}, dtype: {star_img.dtype}, min: {np.min(star_img):.2f}, max: {np.max(star_img):.2f}")
    print(f"Hexagon shape: {hexagon_img.shape}, dtype: {hexagon_img.dtype}, min: {np.min(hexagon_img):.2f}, max: {np.max(hexagon_img):.2f}")
    print(f"Threat shape: {threat_img.shape}, dtype: {threat_img.dtype}, min: {np.min(threat_img):.2f}, max: {np.max(threat_img):.2f}")

    # Example with different size
    large_circle = generate_simple_shape(image_size=(64,64), shape_type='circle', noise_level=0.0)
    # display_image(large_circle, 'Large Circle No Noise')
    print(f"Large circle shape: {large_circle.shape}")

    print("Stimulus generation examples created. Uncomment display_image calls to view.") 