import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def green_cover_app():
    st.title("🌿 Green Cover Analysis")

    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open uploaded image
        image = Image.open(uploaded_file)

        # Display uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert image to NumPy array
        img_array = np.array(image)

        # Check if the image has an alpha channel
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]  # Remove alpha channel if present

        # Reshape image data to a 2D array of pixels
        pixels = img_array.reshape((-1, 3))

        # Apply k-means clustering to classify pixels into 2 clusters (green, non-green)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
        clustered_pixels = kmeans.labels_.reshape(img_array.shape[:2])

        # Determine which cluster corresponds to green areas by comparing cluster centers
        cluster_centers = kmeans.cluster_centers_
        green_cluster = np.argmin(cluster_centers[:, 1])  # Index of the cluster with the least green

        # Create a binary mask for green pixels
        green_mask = (clustered_pixels == green_cluster).astype(np.uint8) * 255

        # Calculate green cover percentage
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_pixels = np.sum(green_mask > 0)
        green_cover_percentage = (green_pixels / total_pixels) * 100
        idle_land_percentage = 100 - green_cover_percentage

        # Display processed image as a binary mask
        processed_image_pil = Image.fromarray(green_mask)

        st.image(processed_image_pil, caption='Processed Image (Green Cover)', use_column_width=True)

        # Display green cover percentage
        st.subheader(f"Green Cover Percentage: {green_cover_percentage:.2f}%")
        st.subheader(f"Idle Land Percentage: {idle_land_percentage:.2f}%")

    else:
        st.write("Please upload an image to analyze.")
