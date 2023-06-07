import cv2
import numpy as np
import torch
import json
import torch.nn as nn
import glob
from pathlib import Path


def createDir(dirPath):
    try:
        path = Path(dirPath)
        path.mkdir(parents=True)
        print("Created Directory!", dirPath)
    except:
        print("Folder already created")
    
def SBP_detection(frame, model_module ):
    img  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 256))#.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)#torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    img /= 255

    img = img.cuda()
    predictions = model_module(img)
    return predictions

def recognize_face(frame, recog_model):
    img  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))#.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)#torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    img /= 255
    #if torch.cuda.is_available:
    img = img.cuda()
    predictions = recog_model(img)
    
    return predictions


def load_labels(filename="utils/FSnameList.json"):
    with open(filename) as f:
        names = json.loads(f.read())
    return names

def AGS_preprocess(frame):
    img  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))#.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)#torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    img /= 255
    return img

def AGS_postprocessing(age_pred, gen_pred, spoof_pred, conf_spoof = 0.5, conf_age= 0.4, conf_gen = 0.5):
    smax = nn.Softmax()
    if spoof_pred == None :
        spoof = None
    else:
        spoof = smax(spoof_pred)[0]
    age_out = smax(age_pred)[0]

    #gmax = nn.Softmax()
    gen_out = smax(gen_pred)[0]
    #print("Spoof Probability: ", age, gen, spoof )
    age = ""
    Gender = ""
    if age_out.data[0] > age_out.data[1] and age_out.data[0] > age_out.data[2] and age_out.data[0] > conf_age:
        Age = "Kid" #+ " - "
    if age_out.data[1] > age_out.data[0] and age_out.data[1] > age_out.data[2] and age_out.data[1] > conf_age:#age_out.data[1] > confidence:#- 0.25 and age_out.data[0] < confidence:
        Age = "Adult" #+ " - "
    if age_out.data[2] > age_out.data[0] and age_out.data[2] > age_out.data[1] and age_out.data[2] > conf_age:
        Age = "Old" #+ " - "

    if gen_out.data[0] > conf_gen:
        Gender =  "Male"
    elif gen_out.data[1] > conf_gen:#- 0.25 and age_out.data[0] < confidence:
        Gender = "Female"

    if gen_out.data[0] > conf_spoof:
        Spoof =  "Live"
    elif gen_out.data[1] > conf_spoof:#- 0.25 and age_out.data[0] < confidence:
        Spoof =  "Spoof"
    else:
        Spoof = None
    #print("Probability: ", Age, Gender, Spoof)
    return {"age": Age, "gen": Gender, "spoof": Spoof }

def get_minIndex(inputlist):
 
    #get the minimum value in the list
    min_value = min(inputlist)
 
    #return the index of minimum value 
    min_index=inputlist.index(min_value)
    return min_index
    
def drawBorder(im, bordersize = 5):
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    
    border = cv2.copyMakeBorder(
        im,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 255, 0]
    )

    return border
def get_maxIndex(inputlist):
 
    #get the minimum value in the list
    min_value = max(inputlist)
 
    #return the index of minimum value 
    min_index=inputlist.index(min_value)
    return min_index

def faceEmbeddingMatching(imgEmbs, dirPath, recog_threshold):
    #print("Matching embedding vectors")
    userList = glob.glob(dirPath + "*.pt")
    #print(userList)
    
    embList = []
    for u in userList:
        #print(u)
        #if embedding_path != None:
        embList.append(u)
    #print(embList)

    #print("number of enlisted persons", len(embList))
    embName = []
    prospectiveIdentityMatrix = []
    for pathLoad in embList:
        #pathLoad = "embVec/sharif.pt"
        #print(pathLoad)
        registeredEmb = torch.load(pathLoad)
        #print("loaded embeddings",registeredEmb.shape)
        embScore = [] 
        
        
        for l in range(imgEmbs.shape[0]):
            
            for re in range(registeredEmb.shape[0]):
                #print(imgEmbs[l].shape, registeredEmb[re].shape)
                distance_image= (imgEmbs[l] - registeredEmb[re]).norm().item()
                #print("distance of images", distance_image)
                embScore.append(distance_image)
        
        averageScore = np.average(embScore)
        identityPerson = pathLoad.split("/")[-1].split(".pt")[0]#.split("_")[-2]
        #print(identityPerson)
        prospectiveIdentityMatrix.append(averageScore)
        #pID = db.session.query(FaceID).get(int(identityPerson)).name
        embName.append(identityPerson)
    #print("listing all available scores", len(embName))
    #for i in range(len(embName)):
        #print(embName[i],prospectiveIdentityMatrix[i] )
    targetIndex = get_minIndex(prospectiveIdentityMatrix)
    

    #print("############################")
    #print("distance between embedding scores (average)", averageScore)
    #print(embName[targetIndex], prospectiveIdentityMatrix[targetIndex])
    #print("############################")
    # read stored embedding vectors
    # Reinitiating Attack List
    #print("****************************attacktype*****************************")
    if prospectiveIdentityMatrix[targetIndex] < recog_threshold:
        return embName[targetIndex]
    else:
        return None
    
def AGS_detection( img, modelagegender, modelspoofing, spoofing = True, ):
        #for imgPath in testImageList:
        #if torch.cuda.is_available:
        img_s = img.cuda()
        img = img.cuda()
        if spoofing == True:
            spoof_pred = modelspoofing(img_s)
            
        else:
            spoof_pred = None
        age_pred, gen_pred = modelagegender(img) 
        # # get the index of the max log-probability
        
        #
        # age_out = smax(y_pred)[0]
        #print("Spoof Probability: ", age_out )

        # if age_out.data[0] > confidence:
        #     return "Live"
        # elif age_out.data[0] > confidence- 0.25 and age_out.data[0] < confidence:
        #     return "Low"
        # else:
        #     return "Spoof"
        return age_pred, gen_pred, spoof_pred


def load_labels(filename="utils/FSnameList.json"):
    with open(filename) as f:
        names = json.loads(f.read())
    return names


#RED = (0, 0, 255)
#GREEN = (0, 255, 0)
#BLUE = (255, 0, 0)
def AGS_prediction(data):
    max_values = {}

    # Iterate over each dictionary in the list
    for dictionary in data:
        # Iterate over each key-value pair in the dictionary
        for key, value in dictionary.items():
            # Check if the key exists in max_values dictionary
            if key in max_values:
                # Update the maximum value for the key if necessary
                if value > max_values[key]:
                    max_values[key] = value
            else:
                # Add the key to max_values dictionary if it doesn't exist
                max_values[key] = value
    print("maximum occurance ",max_values)
    return max_values['age'], max_values['gen'], max_values['spoof']    

def cv_draw_landmark(img_ori, pts, box=None, color=(0, 0, 255), size=1):

    img = img_ori.copy()
    n = pts.shape[1]
    if n <= 106:
        for i in range(n):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1)
    else:
        sep = 1
        for i in range(0, n, sep):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, 1)

    if box is not None:
        left, top, right, bottom = np.round(box).astype(np.int32)
        left_top = (left, top)
        right_top = (right, top)
        right_bottom = (right, bottom)
        left_bottom = (left, bottom)
        cv2.line(img, left_top, right_top, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_top, right_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_bottom, left_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, left_bottom, left_top, BLUE, 1, cv2.LINE_AA)

    return img

def landmark3Ddetection(img, tddfa):
    img = cv2.resize(img, (120,120))
    frame_bgr = img
    xmin, ymin, xmax, ymax = 0 , 0 , 120, 120
    dense_flag = False#args.opt in ('3d',)

    boxes = [[xmin, ymin, xmax, ymax]]#[boxes[0]]
    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

    # refine
    param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
    ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
    res = cv_draw_landmark(frame_bgr, ver)
    return res

