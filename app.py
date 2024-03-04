from flask import Flask, render_template, request

from Final_code import calc

app = Flask(__name__)# interface between my server and my application wsgi

import pickle

model = pickle.load(open(r'C:\Users\Lenovo\Downloads\Diabetes\model.pkl','rb'))


@app.route('/')#binds to an url
def helloworld():
    return render_template("prediction.html")

# @app.route("/prediction", methods=["GET"])
# # def redirect_internal():
# #     return render_template("/prediction.html", code=302)

@app.route('/predicted', methods =['POST'])#binds to an url
def login():
    p =request.form["input1"]
    q= request.form["input2"]
    r= request.form["input3"]
    s= request.form["input4"]
    t= request.form["input5"]
    u= request.form["input6"]
    v= request.form["input7"]
    w= request.form["input8"]
    
    output = model.predict([calc(int(p),int(q),int(r),float(s),float(t),float(u),int(v),int(w))])
    

    result_message = "Not having diabetes" if output[0] == 0 else "Diabetic" 

    return render_template("prediction.html", y = "The person is " + result_message)

#@app.route('/admin')#binds to an url
#def admin():
   # return "Hey Admin How are you?"

if __name__ == '__main__' :
    app.run(debug= False)
    
