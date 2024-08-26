import cv2

def enhance_thermal_image(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    thermal_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Aplicați CLAHE pe imaginea termică
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(thermal_image)

    return thermal_image, enhanced_image

# Încărcați imaginea termică și îmbunătățiți-o
thermal_image_path = 'thermal_image.jpg'
original_thermal_image, enhanced_thermal_image = enhance_thermal_image(thermal_image_path)

# Afișați imaginile
cv2.imshow('Original Thermal Image', original_thermal_image)
cv2.imshow('Enhanced Thermal Image', enhanced_thermal_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
