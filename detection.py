import cv2
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from twilio.rest import Client
import env_vars as ev
from email.message import EmailMessage
import ssl
import smtplib
import time
import threading
import sys

# Load the trained YOLO model
model = YOLO("ML_part/best.pt")

# Variable to track the last time an email was sent
last_email_sent_time = 0

def send_email_with_frame(frame_path, class_name, rounded_conf, location):
    global last_email_sent_time

    try:
        # Function to send email with an attached image
        em = EmailMessage()
        em['From'] = ev.email_s
        em['To'] = ev.email_r
        em['subject'] = ev.subject

        # Create HTML content with formatting
        body = f"""
        <html>
            <body>
                <p>Accident detected of class <b>{class_name}</b> with severity level <b>{rounded_conf}%</b> at location <b>{location}</b>. Check the attached image.</p>
            </body>
        </html>
        """
        em.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(ev.email_s,ev.email_p)
            smtp.sendmail(ev.email_s, ev.email_r, em.as_string())
        
        # Print confirmation message
        print("Email sent!")

        # Update the last email sent time
        last_email_sent_time = time.time()

        return True

    except Exception as e:
        print("Error sending email:", str(e))
        return False

def email_sending_thread(frame_path, class_name, rounded_conf, location):
    global last_email_sent_time

    # Check if 20 seconds have passed since the last email was sent
    elapsed_time = time.time() - last_email_sent_time
    if elapsed_time < 20:
        # If less than 20 seconds have passed, wait for the remaining time
        time.sleep(20 - elapsed_time)

    # Send email
    if send_email_with_frame(frame_path, class_name, rounded_conf, location):
        print("Email sent!")
    else:
        print("Failed to send email")

class Detection:
    detection_result = []

    @staticmethod
    def prediction(image_path):
        # Function to perform object detection using YOLO model
        results = model.predict(source=image_path, show=False)

        for result in results:
            for box in result.boxes:
                class_id = box.cls[0].item()
                cords = [round(x) for x in box.xyxy[0].tolist()]
                conf = round(box.conf[0].item(), 2)
                print("Confidence Score:", conf)  # Print confidence score

                if conf >= 0.5 and class_id == 1 :  # Adjust the confidence threshold as needed
                    print("Object type:", class_id)
                    print("Coordinates:", cords)
                    print("Confidence Score:", conf)
                    print("---")

                    location = "*LOCATION*"
                    class_name = "*MODERATE*" if class_id == 0 else "*SEVERE*"
                    rounded_conf = str(round(conf * 100, 2))

                    # Start a separate thread to send email
                    email_thread = threading.Thread(target=email_sending_thread, args=(image_path, class_name, rounded_conf, location))
                    email_thread.start()

                    # Reinitialize the result list to empty to get updated values for the next detection
                    Detection.detection_result = []
                    Detection.detection_result.append(class_id)
                    Detection.detection_result.append(conf)
                    return Detection.detection_result

        # Return dummy list if nothing detected
        return [0, 0]

    @staticmethod
    def static_detection():
        # Function to perform static image detection
        image_path = "ML_part/temp/temp.jpg"
        Detection.detection_result = Detection.prediction(image_path)
        return Detection.detection_result

    @staticmethod
    def video_stream_detection():
        file_name = sys.argv[1]
        # Function to perform video stream detection
        frame_width, frame_height = 1280, 720
        #stream_url = "http://195.196.36.242/mjpg/video.mjpg"
        stream_url="ML_part/inputs/uploads/"+file_name
        cap = cv2.VideoCapture(stream_url)

        while cap.isOpened():
            # boolean success flag and the video frame
            # if the video has ended, the success flag is False
            is_frame, frame = cap.read()
            if not is_frame:
                break

            resized_frame = cv2.resize(frame, (frame_width, frame_height))

            # Save the resized frame as an image in a temporary directory
            temp_image_path = "ML_part/temp/temp.jpg"
            cv2.imwrite(temp_image_path, resized_frame)

            # Perform object detection on the image and show the results
            model.predict(source=temp_image_path, show=True)
            detection_result = Detection.prediction(temp_image_path)

            # Check for the 'q' key press to quit
            if cv2.waitKey(1) == 0xff & ord('q'):
                break

        # Release the video capture and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

# Call the video_stream_detection method to start the video stream detection
Detection.video_stream_detection()
