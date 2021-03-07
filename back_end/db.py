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
    

def update_user_info(user_name,summary_title):
      firebase = firebase.firebaseApplication('https://summarai-c2de0-default-rtdb.firebaseio.com/', None)
      firebase.put(user_name, summary_title)
      print('Update user ' + user_name + 'with the value of' + summary_title)

def get_user_info(user_name, summary_title):
      firebase = firebase.firebaseApplication('https://summarai-c2de0-default-rtdb.firebaseio.com/', None)
      value = firebase.get(user_name)
      print('Retrieve user ' + user_name + 'with the value of' + summary_title)
      return value
