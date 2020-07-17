# import numpy as np
# from flask import Flask, request, jsonify, render_template
# # from resume_tool import *

# app = Flask(__name__)
# # model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def home():
#     # resume_tool

#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def resume_module():
# # #     '''
# # #     For rendering results on HTML GUI
# # #     '''
# #     file_path = request.form.values()
# # #     final_features = [np.array(int_features)]
# # #     prediction = model.predict(final_features)

# # #     output = round(prediction[0], 2)

# #     return render_template('index.html', resume_tool)

# # @app.route('/predict_api',methods=['POST'])
# # def predict_api():
# #     '''
# #     For direct API calls trought request
# #     '''
# #     data = request.get_json(force=True)
# #     prediction = model.predict([np.array(list(data.values()))])

# #     output = prediction[0]
# #     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def resume_folder():
    # return "Hello"
    # mypath = 'C:/Users/Huleji/Documents/CV and Resume/Sample/' #enter your path here where you saved the resumes
    mypath = input('Enter folder path: ')
    assert os.path.exists(mypath), 'Files not found at '+str(mypath)
    # mypath = open(user_input, 'r+')
    onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return resume_tool(onlyfiles)[0]
    # if request.method == 'POST':
    #     f = request.files['the_file']
    #     f.save('/var/www/uploads/uploaded_file.txt')   

    
if __name__ == "__main__":
    app.run(debug=True)