import logging
import argparse
import cv2
import base64
import glob
import json
import time
import datetime
from fpdf import FPDF
import requests
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, send_file
app = Flask(__name__)

# Function to set up the arguments
def arti_parser(video_path):
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='yolo_l')
    parser.add_argument("-n", "--name", type=str, default='yolox-l', help="model name")

    parser.add_argument(
        "--path", default=f"{video_path}", help="path to video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--app", 
        default="run"
    )
    parser.add_argument(
        "--host=0.0.0.0",
        default=""
    )
    return parser

# Function to turn frames in base64
def encode_image(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None, {"error": "Failed to encode image"}
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8'), None

# Configure logger
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%H:%M:%S')

# OpenAI API key = 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
api_key = ""

prompt = ""

# Function to send the frame to OpenAI API
def interpret_frame_with_openai(frame):
    global prompt
    base64_image, error = encode_image(frame)
    if error:
        return error
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}. Be concise." 
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to interpret image: {response.text}")
        return {"error": response.text, "status_code": response.status_code}

# Function to read frame and use gpt4 to comment
contacted = False
def image_flow_demo_openai_UI_integrated(current_time, args):
    global contacted
    global global_queue_content
    global ind
    print('path', args.path)
    print('demo',  args.demo)
    # if the path has changed aka args.path is not video_source
    path = args.path
    cap = cv2.VideoCapture(path if args.demo == "run" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    INTERPRET_FRAME_INTERVAL = 10 # interpret every 5th frame
    collected_logs = []
    while True:
        print('here1')
        ret_val, frame = cap.read()
        if not ret_val:
            print('restarting LLM')
            cap = cv2.VideoCapture(path if args.demo == "run" else args.camid)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)

        if ret_val:
            print('here1*')
            
            frame_count += 1
            if frame_count % INTERPRET_FRAME_INTERVAL == 0:
                print('here2')
                content = interpret_frame_with_openai(frame)['choices'][0]['message']['content']
                if 'ALERT' in content:
                    # email
                    print('about to send')
                    email_send_dest(your_email, recipient_email, content)
                    contacted = True

                current_time = datetime.now().strftime('%H:%M:%S')
                global_queue_content.append(f"{current_time} :: Camera {lst_sources[args.path]} ::\n {content}")
                logger.info(f"{content}")
                log_message = (current_time, content)
                collected_logs.append(log_message)

                print('values', args.path, video_source)
                if args.path != video_source:
                    break
                print('sources', lst_sources, args.path, video_source)
                yield json.dumps({"text": f"{current_time} :: Camera {lst_sources[video_source]} ::\n {content}"})

LOGO_PATH = 'logo_white.jpg'
LOGO_WIDTH = 10  # Adjust as needed
class PDF(FPDF):
   def header(self):
       self.set_font('Courier', 'B', 12)
       self.cell(0, 10, 'productx Mission Report', 0, 1, 'C')
       # Add logos in the corners of the header
       self.image(LOGO_PATH, 10, 8, LOGO_WIDTH)  # Top-left corner
       self.image(LOGO_PATH, 190, 8, LOGO_WIDTH)  # Top-right corner

   def footer(self):
       self.set_y(-15)
       self.set_font('Courier', 'I', 8)
       self.image(LOGO_PATH, 10, self.get_y(), LOGO_WIDTH)  # Bottom-left corner
       self.image(LOGO_PATH, 190, self.get_y(), LOGO_WIDTH)  # Bottom-right corner
       self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

   def add_content(self, content_list):
       self.set_font('Courier', '', 12)
       for timestamp, content in content_list:
            self.set_font('Courier', 'B', 12)
            self.cell(30, 10, timestamp, ln=0)  # Keeping the timestamp width fixed
            self.set_font('Courier', '', 12)
            self.multi_cell(0, 10, f" {content}")

def imageflow_demo_yolo_UI_integrated(args):
    print('path', args.path, 'demo', args.demo)
    cap = cv2.VideoCapture(args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print('step1 ok')
    if not os.path.exists(args.path):
        print('path does not exist')
    else:
        print('victory')

    if not cap.isOpened():
        print('path exist but cannot be opened')
    else:
        print('victoryy')

    print('about to start')
    while True:
        print('I am reading')
        ret_val, frame = cap.read()

        print('what the heck', ret_val, frame)
        if not ret_val:
            print('restarting CV')
            print('path', args.path, 'demo', args.demo)
            cap = cv2.VideoCapture(args.path if args.demo == "run" else args.camid)
            
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)

        if ret_val:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
    
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    

video_source = None
lst_sources = {}
ind=0
current_time = None
@app.route('/video_feed')
def video_feed():
    global video_source
    global current_time
    global ind

    # Get the URL from the query parameters
    video_source = request.args.get('url')  
    args = arti_parser(video_source).parse_args()
    print('test', args.path, args.experiment_name)
    
    # keep a list of all the new cameras added to the system
    if args.path not in lst_sources:
        lst_sources[args.path]=ind
        ind+=1

    print("passing with", args.path)
    if not args.experiment_name:
        args.experiment_name ="exp_name"

    file_name = os.path.join("exp_dir", args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    logger.info("Args: {}".format(args))
    logger.info("Model Summary: No CV model::reading frames")

    print(args.path)

    if 'rtsp' not in args.path: 
        while not os.path.exists(args.path):
            redirect(url_for('video_feed'))

    return Response(imageflow_demo_yolo_UI_integrated(args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/load_stream', methods=['POST'])
def load_stream():
    global first
    global contacted
    # helpful to restart this so 
    # that I can reload the stream
    contacted = True
    first = True
    print('restarting the LLM cycle')
    # You could include additional validation for the URL here if needed
    return jsonify(success=True)

# Set up the SMTP server
smtp_server = "smtp.gmail.com"
smtp_port = 587
your_email = "jonathanjerabe@gmail.com"
your_password = "ajrn mros lkzm urnu"
recipient_email = ""

@app.route('/prompt', methods=['POST'])
def get_prompt():
    global prompt
    global recipient_email
    data = request.get_json()
    recipient_email = data.get('email')
    prompt = data.get('prompt')
    print(f"prompt received {prompt}")
    if len(prompt)>0:
        return jsonify(success=True)
    return jsonify(success=False)

def email_send_dest(sender, dest, content):
    # Compose the email
    subject = "ASACAM REPORT"
    body = content
    print('hereemail')
    # Create the MIME message
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = dest
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Establish connection to Gmail's SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection

        # Log in to the server
        server.login(your_email, your_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(your_email, recipient_email, text)

        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")

    finally:
        # Close the connection to the server
        server.quit()

# Route to get random text dynamically // THIS IS GONNA BE THE LLM TEXT
first = True
predictor_LLM = None
vis_folder_LLM = None
args_LLM = None
old_source = None
global_queue_content = None
@app.route('/random_text')
def random_text():

    # these values are global because i not only need to initialize
    # them outside the loop, i also need them to conserve the state 
    # of the function.
    global old_source
    global video_source
    global first
    global predictor_LLM
    global vis_folder_LLM
    global args_LLM
    global global_queue_content
    print('LLM is accessed')
    print(video_source)
    if video_source==None:
        redirect(url_for('random_text'))
    print('LLM is running for the first time:', first)
    if first:
        print('LLM is accessing frame')
        old_source = video_source
        args_LLM = arti_parser(video_source).parse_args()

        if 'rtsp' not in args_LLM.path:
            while not os.path.exists(args_LLM.path):
                redirect(url_for('video_feed'))

        if not args_LLM.experiment_name:
            args_LLM.experiment_name = "exp.exp_name"

        file_name = os.path.join("exp_dir", args_LLM.experiment_name)
        os.makedirs(file_name, exist_ok=True)

        logger.info("Args: {}".format(args_LLM))
        logger.info("Model Summary: No CV")

        current_time_LLM = time.localtime()
        old_source = video_source
        first=False
        return Response(image_flow_demo_openai_UI_integrated(current_time_LLM, args_LLM))
    
    # if source has changed and it is not the first time
    if old_source!=video_source:
        first = True
        print('Link has been changed ... LLM relaunching')
        return random_text()
  
    if global_queue_content:
        return Response(json.dumps({"text":global_queue_content[-1]}))
    return Response(json.dumps({"text":"Loading Image Analysis ... "}))

# get the last created directory
def get_last_created_directory(path):
    # Get all subdirectories in the specified path
    subdirs = [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]

    # If there are no directories, return None
    if not subdirs:
        return None

    # Sort subdirectories by creation time (newest first)
    latest_subdir = max(subdirs, key=os.path.getctime)
    
    print(latest_subdir)
    return latest_subdir

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global args_LLM
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # # check if the file exist
        # # put the right there
        # # Save the file
    
        # if exp==None:
        #     return jsonify({'message': 'Error', 'file_path':'None'}), 500
        # if not args_LLM.experiment_name:
        #     args_LLM.experiment_name = exp.exp_name
        # file_name = os.path.join(exp.output_dir, args_LLM.experiment_name, exist_ok)
        # vis_folder = os.path.join(file_name, "vis_res")
        # current_time = time.localtime()
        # print(vis_folder)
        # print('last path guy', get_last_created_directory(vis_folder))
        # save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M", current_time))
        
        # # if the image folder is already created use that instead of creating a new folder
        # image_folder = get_last_created_directory(vis_folder)
        # if save_folder != image_folder and image_folder!=None:
        #     save_folder = image_folder
        #     print('used the older folder')

        # if os.path.exists(save_folder):
        #     print('I think this is it', save_folder)
        # else:
        #     print('Does not exist oops...')
        #     os.makedirs(save_folder, exist_ok=True)

        # save_folder = os.path.join(save_folder, 'report.pdf')
        # print('final folder', save_folder)
        # file.save(save_folder)
        return jsonify({'message': 'PDF saved successfully!', 'file_path':"Testing"}), 200

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mission')
def index():
    return render_template('mission.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
