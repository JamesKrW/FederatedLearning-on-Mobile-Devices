import requests


def get_access_token(text):
    i = 0
    strword = ''
    access_token_str = ''
    isaccessstr = False
    while (i < len(text)):
        if (text[i] == "\"" or text[i] == "\'"):
            while i < len(text):
                i += 1
                if (text[i] != "\"" and text[i] != "\'"):
                    strword += text[i]
                else:
                    break
        if isaccessstr:
            access_token_str = strword
            return strword
        if strword == 'access_token':
            isaccessstr = True
        strword = ''
        i += 1


def get_token():
    url = "http://192.168.42.231:2001/robotservice/auth/login"
    payload = {'password': '92ab1ca81442c05daba8bf59d063e4c9',
               'username': 'gosunyun',
               'grant_type': 'password'}

    files = []
    headers = {'Authorization': 'Basic YWRtaW46YWRtaW4='}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    response_text = response.text.replace(',', '')
    response_text = response_text.replace(':', '')
    response_text = response_text.replace('{', '')
    response_text = response_text.replace('}', '')

    return get_access_token(response_text)


'''
user_id=get_usrid(access_token)
print(user_id)

device_id=get_deviceid(access_token,user_id)
print(device_id)
'''


def speak(strtext, access_token=get_token()):
    url = "http://192.168.42.231:2001/robotservice/device/voiceSoundtextSet.action?access_token=" + access_token
    payload = {'deviceId': 10, 'soundtext': strtext}
    response = requests.request("POST", url, data=payload)
    print(response.text)

speak("你好",get_token())


