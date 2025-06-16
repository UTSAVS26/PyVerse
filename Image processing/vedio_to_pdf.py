import cv2
from fpdf import FPDF
import os
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


global_count=0
clarity = 150
duplicity_threshold = 0.95
desired_fps = 10


def vedio_to_pdf(vedio_path,output_pdf,folder_path):
    global desired_fps
    global duplicity_threshold
    
    
    
    # images_folder = Path(__file__).parent 
    # C:\Users\ACER\Desktop\majorproject2\MedExtract-main\vedio_to_image.pdf

    # # vedio_path = "C:/Users/ACER/Desktop/ALL/medical data extraction/source-code/source-code/4_project_medical_data_extraction/backend/notebooks/WhatsApp Video 2024-04-24 at 2.15.34 PM.mp4"

    # output_pdf = images_folder/'vedio_to_image.pdf' 

    # # folder_path = "C:/Users/ACER/Desktop/ALL/medical data extraction/source-code/source-code/4_project_medical_data_extraction/backend/notebooks/frames"

    # vidcap = cv2.VideoCapture(vedio_path)


    def is_image_clear(image_path, threshold=500):
        img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        global clarity
        if (laplacian_var < clarity):
            return False
        return True

    def delete_png_file(folder_path, filename):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file exists
        if os.path.exists(file_path):
            try:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted {file_path}")
                global global_count
                global_count = global_count - 1
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        else:
            print(f"The file {filename} does not exist in the folder.")




    def calculate_similarity(image1_path, image2_path):
        # Read images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # Check if images were successfully loaded
        if image1 is None or image2 is None:
            return False, 0.0

        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Compute Structural Similarity Index (SSIM) between the two images
        similarity = ssim(gray1, gray2)

        return True, similarity

    def delete_neighbor_duplicates(folder_path, duplicity_threshold):
        # Get list of files in folder
        # files = os.listdir(folder_path)


        # # Iterate through each file
        # for i in range(2,global_count-4):
        #     file1 = files[i]
        #     file2 = files[i + 1]

        #     # Check if both files are images
        #     if file1.lower().endswith(('.png')) and \
        #        file2.lower().endswith(('.png')):

        #         # Calculate similarity between neighboring images
        #         is_valid, similarity = calculate_similarity(os.path.join(folder_path, file1), os.path.join(folder_path, file2))
        #         if is_valid and similarity >= duplicity_threshold:
        #             # Delete file2 if it's a duplicate
        #             os.remove(os.path.join(folder_path, file2))
        #             print(f"Deleted duplicate image: {file2}")
        #             def delete_neighbor_duplicates(folder_path, duplicity_threshold):
        # Get list of files in folder
        files = os.listdir(folder_path)

        # Iterate through each file
        for i in range(len(files) - 1):
            file1 = files[i]
            
            # Check if both files are images
            if file1.lower().endswith('.png') and i + 1 < len(files):  # Ensure i + 1 is within bounds
                file2 = files[i + 1]
                
                # Calculate similarity between neighboring images
                is_valid, similarity = calculate_similarity(os.path.join(folder_path, file1), os.path.join(folder_path, file2))
                if is_valid and similarity >= duplicity_threshold:
                    # Delete file2 if it's a duplicate
                    os.remove(os.path.join(folder_path, file2))
                    print(f"Deleted duplicate image: {file2}")


    def file_exists_in_folder(folder_path, file_name):
        file_path = os.path.join(folder_path, file_name)
        # print('y')
        return os.path.exists(file_path)


    def create_pdf(image_paths, output_path):
        pdf = FPDF()
        pdf.add_page()
        
        for image_path in image_paths:
            pdf.image(image_path, x=None, y=None, w=200)
        
        pdf.output(output_path)


    def delete_images(folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)
        
        # Iterate through each file
        for file in files:
            # Check if the file is an image (ends with .jpg, .png, .gif, etc.)
            if file.endswith(('.png')):
                # Construct the full file path
                file_path = os.path.join(folder_path, file)
                try:
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    def align_document(image_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to read the image.")
            return

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges in the image
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by area and find the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Obtain the corners of the largest contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Ensure that the contour has 4 corners (a rectangle)
        if len(approx) == 4:
            # Rearrange the corners to ensure they are ordered top-left, top-right, bottom-right, bottom-left
            rect = np.array([approx[0][0], approx[1][0], approx[3][0], approx[2][0]], dtype="float32")

            # Define the dimensions of the output image
            width, height = 500, 700
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

            # Calculate the perspective transformation matrix and warp the image
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (width, height))

            # # Display the aligned image
            # cv2.imshow("Aligned Document", warped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            print("Error: The contour doesn't have 4 corners.")



    frame_skip = round(vidcap.get(cv2.CAP_PROP_FPS) / desired_fps)


    success, image = vidcap.read()


    while success:
        # Write the frame to an image file
        global global_count
        cv2.imwrite(r"C:\Users\HP\Desktop\coding_files\python_general\python_project_final\temproraryfolder\frame_%d.png" %global_count, image)
        
        # Skip frames according to the desired FPS
        for _ in range(frame_skip - 1):
            vidcap.grab()
        
        # Read the next frame
        success, image = vidcap.read()
        
        # Increment frame count
        global_count += 1

    print("Frames extracted:", global_count)


    vidcap.release()


    for i in range(1,global_count):
        image_path = r"C:\Users\HP\Desktop\coding_files\python_general\python_project_final\temproraryfolder\frame_%d.png" %i
        # Check if the image is clear
        if is_image_clear(image_path):
            print("Image is clear")
        else:
            filename = r"C:\Users\HP\Desktop\coding_files\python_general\python_project_final\temproraryfolder\frame_%d.png" %i
            delete_png_file(folder_path, image_path)



    global duplicity_threshold
    delete_neighbor_duplicates(folder_path, duplicity_threshold)
    delete_neighbor_duplicates(folder_path, duplicity_threshold)
    delete_neighbor_duplicates(folder_path, duplicity_threshold)
    delete_neighbor_duplicates(folder_path, duplicity_threshold)

    images = []  

    for i in range(global_count - 2):
        file_name = "frame_%d.png" % i
        if file_exists_in_folder(folder_path, file_name):
            images.append(os.path.join(folder_path, file_name))






    # for i in range(global_count - 2):
    #     file_name = "frame_%d.png" % i
    #     if file_exists_in_folder(folder_path, file_name):
    #         align_document("C:/Users/ACER/Desktop/ALL/medical data extraction/source-code/source-code/4_project_medical_data_extraction/backend/notebooks/frames/frame_%d.png" %i )


    create_pdf(images, output_pdf)
    delete_images(folder_path)

