import sys
import os
import requests
import logging

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    logging.info("Downloading "+id + " to "+destination)
    logging.info("Please be patient, it may take a while...")
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    logging.info("...")
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    
    logging.info("Done with " + id)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

'''
    to download the .t7 NTS models used for text simplification
    if for some reason, this doanload fails, please use the direct urls:
    - for NTS:
        https://drive.google.com/open?id=0B_pjS_ZjPfT9dEtrbV85UXhSelU
    -for NTS-w2v:
        https://drive.google.com/open?id=0B_pjS_ZjPfT9ZTRfSFp4Ql92U0E
'''

if __name__ == "__main__":
    try:
        out_dir = sys.argv[1]
        logging.info("Saving files to: " + out_dir)
    except:
        out_dir = os.path.dirname(os.path.realpath(__file__))
        logging.info("Saving files to: " + out_dir)
        
    #NTS_model = '0B_pjS_ZjPfT9dEtrbV85UXhSelU' 
    #NTS_model_output = 'NTS_epoch11_10.19.t7'
    #download_file_from_google_drive(NTS_model, os.path.join(out_dir, NTS_model_output))

    #NTS_w2v_model = '0B_pjS_ZjPfT9ZTRfSFp4Ql92U0E' 
    #NTS_w2v_model_output = 'NTS-w2v_epoch11_10.20.t7'
    #download_file_from_google_drive(NTS_w2v_model, os.path.join(out_dir, NTS_w2v_model_output))

    #model = '1_JsQ_iMnHwvnyd5vZM-6BMqe9hnVqrPi'
    model = '19QDCd4UMgt3FtlYYwu0qZU3G1F9_XCvk'
    download_file_from_google_drive(model, 'paraphrase-model.tar.gz')

