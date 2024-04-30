from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def homepage():
    UPLOAD_FOLDER = "ML_part/inputs/uploads/"
    if request.method == "POST":
        f = request.files["uploadedImage"]
        file_name = secure_filename(f.filename)
        f.save(f"ML_part/inputs/uploads/{file_name}")
        print("Uploaded file:", file_name)

        # Execute a.py and pass the uploaded filename as an argument
        subprocess.run(["python", "ML_part/detection.py", file_name])


    return render_template("home.html")

if __name__ == "__main__":
    app.run(port=1234, debug=True)
