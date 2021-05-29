import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import send_from_directory

classes = ["rare","welldone"] #분류할 class명을 정의

num_classes = len(classes) #list 길이를 취득하고 정의

image_size = 50 #이미지 사이즈를 정의

#UPLOAD_FOLDER는 업로드된 파일을 격납할 장소를 지정 
UPLOAD_FOLDER = "uploads"

#업로드를 허가할 파일을 지정
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif']) 

#Flask 클래스의 메소드를 사용할 수 있도록 함. 例：@app.route、app.run()
app = Flask(__name__) 

#파일 명이 올바른 포멧에 되어 있는지 확인
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

#학습된 모델을 로드함
model = load_model('my_model.h5')

#URL를 받은 HTTP메소드를 지정
@app.route('/', methods=['GET', 'POST']) 
def upload_file():
    if request.method == 'POST': #request.method는 페이지에 접속 방식을 탐지하는 기능. POST여부를 판별함.
        if 'file' not in request.files: #파일이 없는 경우
            flash('ファイルがありません')
            return redirect(request.url) # request된 페이지에 전송
        file = request.files['file']
        if file.filename == '': #파일 명이 없는 겨우　　
            flash('ファイルがありません')
            return redirect(request.url) # request된 페이지에 전송
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) #sanitize
            file.save(os.path.join(UPLOAD_FOLDER, filename)) #파일 저장
            filepath = os.path.join(UPLOAD_FOLDER, filename) #저장처를 filepath에 격납함

            #받은 이미지를 읽어 np형식으로 변환
            img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])

            #변환한 데이터를 모델에 넘겨 예측함
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "당신의 고기는 " + classes[predicted] + " 입니다!"

            #render_template 인수에 answer=pred_answer 와 전달함으로써 index.html 에 작성한 answer 에 pred_answer 를 대입
            #이 인수에 전달할 html 파일은 temlpates 폴더에 놓아 둘 필요가 있음
            return render_template("index.html",file = file,answer=pred_answer) 
    #POST 요청이 이루어지지 않을 때는 index.html의 answer에는 아무것도 표시하지 않음
    return render_template("index.html",answer="") 

@app.route('/uploads/<filename>')
def uploaded_file(filename): # 파일을 표지
    return send_from_directory(UPLOAD_FOLDER, filename)

    
@app.route('/AI_2021_flask-master/templates/home1.html') 
def link1():
        return render_template('home1.html')

#Python 스크립트를 실행할 때만 if __name__== '_main_' : 아래의 처리를 실행시키도록 함

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080)) #os. environ.get 에서 취득한 환경변수의 문자열형(str) 값을 수치형(int)으로 변경하여 대입
    app.run(host ='0.0.0.0',port = port) #app. run()이 실행되고 서버가 가동
