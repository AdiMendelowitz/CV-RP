import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import get_affine_transform, get_perspective_transform, warp_affine, warp_perspective, rotate, resize


def test_affine():
    """Test affine transformation"""

    img = np.zeros((200, 200), dtype='uint8')
    img[50:150, 50:150] = 255

    # Define transormation: rotate square
    scr_pts = np.array([[50, 50], [150, 50], [150, 150]], dtype='float32')
    dst_pts = np.array([[75, 50], [150, 75], [50, 150]], dtype='float32')

    # Get transformation matrix
    M = get_affine_transform(scr_pts, dst_pts)
    M_cv = cv2.getAffineTransform(scr_pts, dst_pts)

    print("My affine matrix:\n", M)
    print("\nOpenCV affine matrix:\n", M_cv)
    print("\nDifference:\n", np.abs(M - M_cv).max())

    # Apply transformation
    warped = warp_affine(img, M, output_shape=(200, 200))
    warped_cv = cv2.warpAffine(img, M_cv, dsize=(200, 200))

    # Visualize
    fix, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(warped, cmap='gray')
    axes[1].set_title('My Affine Warp')
    axes[2].imshow(warped_cv, cmap='gray')
    axes[2].set_title('OpenCV Affine Warp')
    plt.tight_layout()
    plt.savefig('affine_test.png')
    plt.show()

def test_perspective():
    """Test perspective transformation"""

    img = np.zeros((200, 200), dtype='uint8')
    img[50:150, 50:150] = 255

    # Define transormation: rotate square
    scr_pts = np.array([[50, 50], [250, 50], [250, 250], [50, 250]], dtype='float32')
    dst_pts = np.array([[80, 50], [220, 50], [250, 250], [50, 250]], dtype='float32')

    # Get transformation matrix
    H = get_perspective_transform(scr_pts, dst_pts)
    H_cv = cv2.getPerspectiveTransform(scr_pts, dst_pts)

    print("My perspective matrix:\n", H)
    print("\nOpenCV perspective matrix:\n", H_cv)
    print("\nDifference:\n", np.abs(H - H_cv).max())

    # Apply transformation
    warped = warp_perspective(img, H, output_shape=(300, 300))
    warped_cv = cv2.warpPerspective(img, H_cv, dsize=(300, 300))

    # Visualize
    fix, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(warped, cmap='gray')
    axes[1].set_title('My Perspective Warp')
    axes[2].imshow(warped_cv, cmap='gray')
    axes[2].set_title('OpenCV Perspective Warp')
    plt.tight_layout()
    plt.savefig('perspective_test.png')
    plt.show()

def test_rotation():
    """Test rotation transformation"""

    img = np.zeros((200, 200), dtype='uint8')
    img[50:150, 50:150] = 255

    # Rotate by 45 degrees
    rotated = rotate(img, angle_degrees=45, center=(100, 100))
    M_cv = cv2.getRotationMatrix2D((100, 100), 45, 1.0)
    rotated_cv = cv2.warpAffine(img, M_cv, (200, 200))

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(rotated, cmap='gray')
    axes[1].set_title('My Rotation')
    axes[2].imshow(rotated_cv, cmap='gray')
    axes[2].set_title('OpenCV Rotation')
    plt.tight_layout()
    plt.savefig('rotation_test.png')
    plt.show()

if __name__ == '__main__':
    print("Testing Affine Transformation:")
    test_affine()

    print("\nTesting Perspective Transformation:")
    test_perspective()

    print("\nTesting Rotation:")
    test_rotation()