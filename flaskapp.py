from flask import Flask, render_template,request,redirect,Response
import os
from werkzeug.utils import secure_filename
import process as p
import cv2
import time
app = Flask(__name__,static_folder='static')


@app.route("/")
@app.route("/home")
def homepage():
	return render_template("upload.html")



def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture('./static/abc.mp4')

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




app.config["FILE_UPLOADS"] = "./data/video/"
app.config["ALLOWED_EXTENSIONS"] = ["MP4","AVI"]

def allowed(filename):

	if not "." in filename:
		return False

	ext = filename.rsplit(".",1)[1]

	if ext.upper() in app.config["ALLOWED_EXTENSIONS"]:
		return True
	else:
		return False






@app.route("/upload", methods=["GET", "POST"])
def upload():
	if request.method == "POST":

		if request.files:

			video = request.files["image"]

			if video.filename == "":
				print("The file must have a name")
				return redirect(request.url)

			if not allowed(video.filename):
				print("The given extension is not allowed")
				return redirect(request.url)



			else:
				filename = secure_filename(video.filename)
				video.save(os.path.join(app.config["FILE_UPLOADS"], filename))



			print("video saved")
			r0=p.processvid(filename)

			#return redirect(request.url)

	return render_template("result.html",r1=r0)


@app.route("/ppp")
def new():
	return render_template("new.html")

# 	pass





if __name__ == '__main__':
	app.run(debug=True)
