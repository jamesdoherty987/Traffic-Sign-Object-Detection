from dotenv import load_dotenv
load_dotenv()

import kagglehub

path = kagglehub.competition_download('traffic-sign-object-detection-challenge')
print("Path to competition files:", path)