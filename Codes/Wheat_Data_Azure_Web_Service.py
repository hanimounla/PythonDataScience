import urllib2
# If you are using Python 3+, import urllib instead of urllib2

import json 



column_names = ["id", "area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "wheat_type"]
values = ['','13.74','14.05','0.8744','5.482','3.114','2.932','4.825','']
data = {
        "Inputs": {
                "Single_wheat_parametes_input":
                    {
                    "ColumnNames": column_names ,
                    "Values": [ values, ]
                    },        
                  },
            "GlobalParameters": {}
       }

body = str.encode(json.dumps(data))

url = 'https://ussouthcentral.services.azureml.net/workspaces/264711fbdc9d48ba8205ed385fce483e/services/050153fa554e4b5786d21ae02389e276/execute?api-version=2.0&details=true'
api_key = 'VlDmE4z0FVLayQqOYjcslYbGVUi7uM5kxTzcgVj9aAyXM1s455mZ4a3Vdpi3xfLNwIIYBIlgFEBdj66eX9dsjg==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers) 

try:
    response = urllib2.urlopen(req)

    # If you are using Python 3+, replace urllib2 with urllib.request in the above code:
    # req = urllib.request.Request(url, body, headers) 
    # response = urllib.request.urlopen(req)

    result = response.read()
    wjdata = json.loads(result)
    predicted_wheat_type = wjdata['Results']['Predected_Wheat_type_output']['value']['Values'][0][0]
    print "The Predicted Wheat type:", predicted_wheat_type
       
except urllib2.HTTPError, error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())

    print(json.loads(error.read()))
    
    