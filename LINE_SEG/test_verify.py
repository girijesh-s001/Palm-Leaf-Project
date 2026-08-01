import line_seg
import os

if __name__ == "__main__":
    img_path = r"c:\Users\Girijesh\Downloads\LINE_SEG\input-6.jpeg"
    if os.path.exists(img_path):
        print(f"Running verification on {img_path}")
        # Note: process_image uses cv2.imshow, which might fail or hang in a headless environment.
        # However, it also saves images to the 'output' directory.
        # To avoid blocking, we can monkeypatch cv2.imshow and cv2.waitKey if needed.
        import cv2
        cv2.imshow = lambda *args, **kwargs: None
        cv2.waitKey = lambda *args, **kwargs: 0
        
        line_seg.process_image(img_path)
        print("Verification run completed.")
    else:
        print(f"Error: {img_path} not found.")
