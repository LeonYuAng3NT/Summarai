from firebase import firebase 
firebase = fireabase.FirebaseApplication('https://')
result = firebase.get('users')

config = {
  "apiKey": "apiKey",
  "authDomain": "summarai-c2de0.firebaseapp.com",
  "databaseURL": "https://summarai-c2de0-default-rtdb.firebaseio.com/",
  "storageBucket": "c2de0.appspot.com",
  "serviceAccount": "path/to/serviceAccountCredentials.json"
}

def db_authentication():
    firebase = fireabase.FirebaseApplication('https://summarai-c2de0-default-rtdb.firebaseio.com/')
    authentication = firebase.FirebaseAuthentication('suummarai','leonzhang1996@hotmail.com',extra={'id': 123})
    result = firebase.get('/users', None, {'print':'pretty'})
    


