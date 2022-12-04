doclogin:
	cat ~/.docker/access_token | docker login --username mm12063 --password-stdin

iclogin:
	ibmcloud login -a cloud.ibm.com -g 2022-fall-student-mm12063 -r us-south --apikey @~/.bluemix/apiKey.json
	ibmcloud cr login
