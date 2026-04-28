import os
import cv2 as cv
import numpy as np


IMAGE_FILES = [
    "100-0023_img.jpg",  # left
    "100-0024_img.jpg",  # center
    "100-0025_img.jpg",  # right
]


def load_images():
    images = []

    for filename in IMAGE_FILES:
        img = cv.imread(filename)

        images.append(img)

    return images


def show_resized(window_name, img, max_width=900):
    h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        img = cv.resize(img, None, fx=scale, fy=scale)

    cv.imshow(window_name, img)

def detect_and_match(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # SIFT 특징점 검출
    detector = cv.SIFT_create()

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    print(f"[INFO] keypoints1: {len(keypoints1)}")
    print(f"[INFO] keypoints2: {len(keypoints2)}")

    # SIFT는 실수형 descriptor이므로 BruteForce L2 사용
    matcher = cv.DescriptorMatcher_create("BruteForce")

    # knnMatch: 가장 가까운 후보 2개를 찾음
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"[INFO] good matches: {len(good_matches)}")

    return keypoints1, keypoints2, good_matches

def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        raise ValueError("Homography 계산에는 최소 4개의 매칭점이 필요합니다.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, inlier_mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography 계산 실패")

    inlier_count = int(inlier_mask.sum())
    print(f"[INFO] inliers: {inlier_count} / {len(matches)}")

    return H, inlier_mask

def get_image_corners(img):
    h, w = img.shape[:2]

    corners = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)
    return corners

def transform_corners(img, H):
    corners = get_image_corners(img)
    transformed = cv.perspectiveTransform(corners, H)
    return transformed

def compute_canvas(left, center, right, H_left_to_center, H_right_to_center):
    center_corners = get_image_corners(center)

    left_corners = transform_corners(left, H_left_to_center)
    right_corners = transform_corners(right, H_right_to_center)

    all_corners = np.vstack((
        left_corners,
        center_corners,
        right_corners
    ))

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translate_x = -x_min
    translate_y = -y_min

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    T = np.array([
        [1, 0, translate_x],
        [0, 1, translate_y],
        [0, 0, 1]
    ], dtype=np.float64)

    print(f"[INFO] canvas size: {canvas_width} x {canvas_height}")
    print(f"[INFO] translation: ({translate_x}, {translate_y})")

    return canvas_width, canvas_height, T

def blend_images(base, overlay):
    mask_base = np.any(base > 0, axis=2)
    mask_overlay = np.any(overlay > 0, axis=2)

    overlap = mask_base & mask_overlay

    result = base.copy()

    # 단순 덮어쓰기 영역
    result[~mask_base & mask_overlay] = overlay[~mask_base & mask_overlay]

    # 겹치는 영역 → 평균 blending
    result[overlap] = (
        0.5 * base[overlap] + 0.5 * overlay[overlap]
    ).astype(np.uint8)

    return result


def stitch_images(left, center, right, H_left_to_center, H_right_to_center, canvas_width, canvas_height, T):
    H_left_canvas = T @ H_left_to_center
    H_center_canvas = T
    H_right_canvas = T @ H_right_to_center

    warped_left = cv.warpPerspective(left, H_left_canvas, (canvas_width, canvas_height))
    warped_center = cv.warpPerspective(center, H_center_canvas, (canvas_width, canvas_height))
    warped_right = cv.warpPerspective(right, H_right_canvas, (canvas_width, canvas_height))

    result = warped_center.copy()
    result = blend_images(result, warped_left)
    result = blend_images(result, warped_right)

    return result

def main():
    left, center, right = load_images()

    # left-center 매칭
    kp_left, kp_center1, matches_left_center = detect_and_match(left, center)

    matched_left_center = cv.drawMatches(
        left, kp_left,
        center, kp_center1,
        matches_left_center[:100],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv.imwrite("matches_left_center.jpg", matched_left_center)
    show_resized("left-center matches", matched_left_center)

    # center-right 매칭
    kp_center2, kp_right, matches_center_right = detect_and_match(center, right)

    matched_center_right = cv.drawMatches(
        center, kp_center2,
        right, kp_right,
        matches_center_right[:100],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv.imwrite("matches_center_right.jpg", matched_center_right)
    show_resized("center-right matches", matched_center_right)

    H_left_to_center, mask_left_center = compute_homography(
        kp_left,
        kp_center1,
        matches_left_center
    )

    H_center_to_right, mask_center_right = compute_homography(
        kp_center2,
        kp_right,
        matches_center_right
    )

    H_right_to_center = np.linalg.inv(H_center_to_right)

    canvas_width, canvas_height, T = compute_canvas(
        left,
        center,
        right,
        H_left_to_center,
        H_right_to_center
    )

    stitched = stitch_images(
        left,
        center,
        right,
        H_left_to_center,
        H_right_to_center,
        canvas_width,
        canvas_height,
        T
    )

    cv.imwrite("stitched_result.jpg", stitched)
    show_resized("stitched result", stitched, max_width=1200)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()