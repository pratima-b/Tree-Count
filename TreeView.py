import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas
import time

class ImageSeg:
    def __init__(self, img):
        self.img = np.array(img)

    def find_red_marks(self):
        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Define color range for detecting red marks
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask for the red color
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of the red marks
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def count_red_marks(self):
        contours = self.find_red_marks()
        return len(contours)

    def mark_red_marks(self):
        contours = self.find_red_marks()
        marked_img = np.copy(self.img)
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw red rectangle
            cv2.putText(marked_img, f'Tree {i+1}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return marked_img

    def color_filter(self):
        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Define color range for detecting trees (green color)
        lower_bound = np.array([30, 40, 40])  # Lower bound for green color in HSV
        upper_bound = np.array([90, 255, 255])  # Upper bound for green color in HSV

        # Create a mask for the specified color range
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        # Apply the mask to the original image
        filtered_img = cv2.bitwise_and(self.img, self.img, mask=mask)

        return filtered_img

    def preprocess_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)
        return edges

    def post_process(self, edge_img):
        # Apply morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
        return opened_img

    def count_trees_without_red_marks(self):
        filtered_img = self.color_filter()
        edge_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(edge_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        return num_labels - 1  # Subtract 1 to exclude the background

    def mark_trees_without_red_marks(self):
        filtered_img = self.color_filter()
        edge_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(edge_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        marked_img = np.copy(self.img)

        for i in range(1, num_labels):  # Skip the background label
            x, y, w, h, _ = stats[i]
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw red rectangle
            cv2.putText(marked_img, f'Tree {i}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return marked_img

    def count_trees(self):
        red_mark_count = self.count_red_marks()
        if red_mark_count > 0:
            return red_mark_count
        else:
            return self.count_trees_without_red_marks()

    def mark_trees(self):
        red_mark_count = self.count_red_marks()
        if red_mark_count > 0:
            return self.mark_red_marks()
        else:
            return self.mark_trees_without_red_marks()

    def crop_image(self, x_start, y_start, width, height):
        # Crop the image based on the provided coordinates
        cropped_img = self.img[y_start:y_start+height, x_start:x_start+width]
        return cropped_img

st.set_page_config(page_title="Tree View", page_icon="img.svg")

# Streamlit Web App
st.title("🌳 TreeView - Tree Enumeration App")
st.write("**Upload an image, and the app will count and mark the trees in it!**")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image with PIL and convert to OpenCV format
    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.write("### Uploaded Image:")
    # Using columns to display images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Original Image', use_column_width=True)

    # Simulate a short processing delay to show spinner
    with st.spinner("Processing image..."):
        time.sleep(2)  # Simulated delay

    # Create ImageSeg object
    obj = ImageSeg(image)

    # Let user choose to crop or use the full image
    crop_option = st.radio(
        "Would you like to crop the image or use the full image?",
        ("Use Full Image", "Crop Image")
    )

    if crop_option == "Crop Image":
        st.write("### Crop the Image")

        # Use `st_canvas` to allow user to draw a rectangle over the image
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill color for the crop box
            stroke_width=2,
            stroke_color="red",
            background_image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=image.shape[0],
            width=image.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            # Extract rectangle coordinates from the drawn box
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                # Get the rectangle coordinates
                obj_coords = objects[0]
                x_start = int(obj_coords["left"])
                y_start = int(obj_coords["top"])
                width = int(obj_coords["width"])
                height = int(obj_coords["height"])

                # Crop the image
                cropped_img = obj.crop_image(x_start, y_start, width, height)
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                st.image(cropped_img_rgb, caption="Cropped Image", use_column_width=True)

                # Use the cropped image for further processing
                obj = ImageSeg(cropped_img)
    else:
        st.write("Using the full image for processing.")

    # Add progress bar to simulate processing
    st.write("Counting and marking trees...")
    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.03)  # Simulating processing step
        progress.progress(i + 1)

    # Count and mark trees (whether cropped or full image)
    final_count = obj.count_trees()
    marked_img = obj.mark_trees()

    # Convert OpenCV image to RGB for display in Streamlit
    marked_img_rgb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.image(marked_img_rgb, caption=f"Processed Image - {final_count} Trees", use_column_width=True)

    # Allow user to download the processed image
    img_pil = Image.fromarray(marked_img_rgb)  # Convert to PIL for saving
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Processed Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png"
    )

    # Display the final tree count
    st.write(f"🌲 **Final Estimated Tree Count: {final_count} trees**")
