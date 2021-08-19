from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import csv
app = Flask(__name__, static_url_path='/static')



@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

@app.route('/comp1')
def parse():
    return app.send_static_file('comp1.html')

@app.route('/data')
def parse1():
	return render_template('data.html')
	
@app.route('/detect')
def parse2():
	return render_template('detect.html')
	
@app.route('/similarity')
def parse3():
	return render_template('similarity.html')
		
@app.route('/success', methods = ['POST'])  
def success():  
	if request.method == 'POST':
		#f = request.files['fileToUpload'].read()
		f = request.files['fileToUpload']
		if f.filename == '':
			print('no filename')
			return redirect(request.url)
		else:
			print('data/files'+f.filename)
			if os.path.isfile('data/files/'+f.filename):
				print('I am in asc')
				with open('data/cache.csv', 'r') as csvfile:
					csvreader = csv.reader(csvfile)
					for row in csvreader:
						if row[0]==f.filename:
							final = row[1]
			else:
				f.save('data/files/'+f.filename)
				import train as tr
				final = tr.getfile(f.filename)
			print("done")
			return render_template("success.html", name = final)  
		 
@app.route('/success1', methods = ['POST'])  
def success1():  
	if request.method == 'POST':
		f1 = request.files['fileToUpload1']
		f2 = request.files['fileToUpload2']
		if f1.filename == '':
			print('no filename')
			return redirect(request.url)
		elif f2.filename == '':
			print('no filename')
			return redirect(request.url)
		else:
			f1.save('data/simi/files/'+f1.filename)
			f2.save('data/simi/files/'+f2.filename)
			import simi as sm
			val = sm.calculate(f1.filename, f2.filename)
			print("done")
			return render_template("success.html", name = val)  	
if __name__ == '__main__':
    app.debug = True
    app.run()
