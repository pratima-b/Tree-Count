import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time
from streamlit_drawable_canvas import st_canvas

class ImageSeg:
    def __init__(self, img):
        self.img = np.array(img)

    def find_red_marks(self):
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
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
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(marked_img, f'Tree {i+1}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return marked_img

    def count_trees_without_red_marks(self):
        return 0  # Simplified for demo purposes

    def count_trees(self):
        red_mark_count = self.count_red_marks()
        return red_mark_count

    def mark_trees(self):
        return self.mark_red_marks()

# TreeView app function
def treeview_app():
    st.title("🌳 TreeView - Tree Enumeration App")
    st.write("**Upload an image, crop it if needed, and the app will count and mark the trees in it!**")

    uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the image with PIL and convert to OpenCV format
        image = Image.open(uploaded_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Display the original image
        st.write("### Uploaded Image:")
        st.image(image, caption='Original Image', use_column_width=True)

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
                    cropped_img = image[y_start:y_start + height, x_start:x_start + width]
                    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    st.image(cropped_img_rgb, caption="Cropped Image", use_column_width=True)

                    # Use the cropped image for further processing
                    obj = ImageSeg(cropped_img)
        else:
            st.write("Using the full image for processing.")
            obj = ImageSeg(image)

        # Simulate a short processing delay to show spinner
        with st.spinner("Processing image..."):
            time.sleep(2)  # Simulated delay

        # Count and mark trees (whether cropped or full image)
        final_count = obj.count_trees()
        marked_img = obj.mark_trees()

        # Convert OpenCV image to RGB for display in Streamlit
        marked_img_rgb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)

        st.write("### Processed Image:")
        st.image(marked_img_rgb, caption=f"Processed Image - {final_count} Trees", use_column_width=True)

        # Display the final tree count
        st.write(f"🌲 **Final Estimated Tree Count: {final_count} trees**")
