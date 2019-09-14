# Importing all necessary libraries 
import cv2 
import os 
import options

# from tqdm import tqdm

def extractVideoFrames(**kwargs):
    
    assert kwargs.keys is not None , "Make sure you give filename of the video with the absolute path"
    
    # Read the video from specified path 
    filename = kwargs.get("filename")
    cam = cv2.VideoCapture(filename) 
    
    try: 
        # creating a folder named data 
        output_folder = kwargs.get("output_folder") if "output_folder" in kwargs.keys() else "./"
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder)
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
    
    # frame 
    currentframe = 0
    
    while(True): 
        
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # if video is still left continue creating images 
            print(output_folder)
            name = os.path.join(output_folder,"{}_frame_{}.jpg".format(filename.split("/")[-1].split(".")[0] , currentframe))
            print ('Creating frame ...' + name) 
    
            # writing the extracted images 
            #io.imsave(name, frame)
            cv2.imwrite(name, frame) 
    
            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    args = options.parseArguments()
    extractVideoFrames(filename=args.input_dir, output_folder=args.output_dir)