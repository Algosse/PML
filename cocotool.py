import numpy as np
from matplotlib import image as Image
import matplotlib.pyplot as plt
import json

class COCO:
    
    def __init__(self, rootPath):
        self.rootPath = rootPath
        self.annotations = {
            'train': {
                'captions': None,
                'instances': None,
                'person_keypoints': None
            },
            'val': {
                'captions': None,
                'instances': None,
                'person_keypoints': None
            }
        }
        self.categories = None
        self.superCategories = None
            
    def loadAnnotationFile(self, typeData = "train", typeAnnotation="instances"):
        
        assert typeData in ["train", "val"], "typeData has to be either 'train' or 'val'"
        assert typeAnnotation in ["captions", "instances", "person_keypoints"], 'typeAnnotation has to be either "captions", "instances" or "person_keypoints"'
        
        fileName = f"{typeAnnotation}_{typeData}2017.json"
            
        if self.annotations[typeData][typeAnnotation] == None:   
            print(f"Loading {fileName}")

            with open(self.rootPath + "/annotations/"  + fileName, 'r') as f:
                data = json.load(f)
                self.annotations[typeData][typeAnnotation] = data
                print(f"{fileName} loaded")
                return data
            
        else:
            #print(f"{fileName} already loaded")
            return self.annotations[typeData][typeAnnotation]
    
    def loadInstancesData(self):
        # Load train and test (=val) instances data
        
        trainData = self.instancesToData(typeData = "train")
        testData = self.instancesToData(typeData = "val")
        
        return (trainData, testData)
        
    def instancesToData(self, typeData="train"):
        """
            Generator. Convert the coco instances dataset    
        """

        assert typeData in ['train', 'val'], "typeData has to be either 'train' or 'val'"
        
        annotations = self.loadAnnotationFile(typeData = typeData, typeAnnotation = "instances")
        
        # Usefull to retrieve image or class by id
        images = {i['id']: i for i in annotations['images']}
        classes = {i['id']: i for i in annotations['categories']}

        for annotation in annotations["annotations"]:
            result = annotation.copy()

            idCategory = annotation['category_id']
            result['super_category'] = classes[idCategory]['supercategory']
            result['category_name'] = classes[idCategory]['name']

            imageData = images[annotation['image_id']]
            image_path = self.rootPath + f"/images/{typeData}2017/" + imageData['file_name']

            im = Image.imread(image_path)
            result['image'] = np.asarray(im)

            # Maybe change the result in a form usefull for a model. Like create a target key.
            yield result
    
    def getImageByFileName(self, fileName = '391895.jpg'):
        
        # fileName always have the format 000000000016.jpg
        fileName = (fileName.split('.')[0] + ".jpg").zfill(16)

        # Try train data
        trainAnnotations = self.loadAnnotationFile(typeData = "train", typeAnnotation = "instances")
        for image in trainAnnotations['images']:
            if image['file_name'] == fileName:
                return self.getImageById(image['id'], "train")
        
        # Try test data
        trainAnnotations = self.loadAnnotationFile(typeData = "val", typeAnnotation = "instances")
        for image in trainAnnotations['images']:
            if image['file_name'] == fileName:
                return self.getImageById(image['id'], "val")
        
        print("Warning: Image not found")
        return None
    
    def getImageById(self, id = 391895, typeData = 'train'):
        
        annotations = self.loadAnnotationFile(typeData = typeData, typeAnnotation = "instances")

        imageData = None
        for image in annotations['images']:
            if image['id'] == id:
                imageData = image.copy()
                break
        
        if imageData == None:
            raise ValueError(f'id not found in {typeData} dataset')

        # Fill imageData with every existing instances
        imageData['instances'] = []
        for instance in annotations['annotations']:
            if instance['image_id'] == id:
                imageData['instances'].append(instance.copy())

                # Add category to each element on the picture
                for category in annotations['categories']:
                    if  imageData['instances'][-1]['category_id'] == category['id']:
                        imageData['instances'][-1]['category'] = category
        
        captions = self.loadAnnotationFile(typeData=typeData, typeAnnotation='captions')
        for caption in captions['annotations']:
            if caption['image_id'] == imageData['id']:
                imageData['caption'] = caption
        
        # Add image to imageData
        im = Image.imread(self.rootPath + f"/images/{typeData}2017/" + imageData['file_name'])
        imageData['image'] = im

        return imageData

    def showImageInstancesSegmentation(self, imageData, index = None, minArea = 2000, figsize = (12,12)):

        plt.figure(figsize = figsize)

        if index != None:
            self.showSegmentation(imageData['instances'][index])
        else:
            for instance in imageData['instances']:
                if instance['area'] > minArea:
                    self.showSegmentation(instance)
            
        plt.imshow(imageData['image'])
        plt.legend()
        plt.title(imageData['caption']['caption'])

    def showSegmentation(self, instance):
        x = instance['segmentation'][0][::2]
        y = instance['segmentation'][0][1::2]

        x.append(x[0])
        y.append(y[0])
        if ("category" in instance):
            plt.plot(x,y, label = instance['category']['name'])
        else:
            plt.plot(x,y)
    

    def showImageInstancesBbox(self, imageData, index = None, minArea = 2000, figsize = (12,12)):
        plt.figure(figsize = figsize)

        if index != None:
            instance = imageData['instances'][index]
            self.showBbox(instance)
        else:
            for instance in imageData['instances']:
                if instance['area'] > minArea:
                    self.showBbox(instance)
            
        plt.imshow(imageData['image'])
        plt.legend()
        plt.title(imageData['caption']['caption'])

    def showBbox(self, instance):
        # Print the box around the object
        origin = (instance['bbox'][0], instance['bbox'][1])
        width = instance['bbox'][2]
        height = instance['bbox'][3]

        xR = [origin[0], origin[0], origin[0] + width, origin[0] + width, origin[0]]
        yR = [origin[1], origin[1] + height, origin[1] + height, origin[1], origin[1]]

        if ("category" in instance):
            plt.plot(xR,yR, label = instance['category']['name'])
        else:
            plt.plot(xR,yR)

    def getCategories(self):

        if self.categories != None:
            return self.categories
        
        annotations = self.loadAnnotationFile(typeData = 'train', typeAnnotation = "instances")

        categories = [category['name'] for category in annotations['categories']]

        self.categories = categories
        return categories

    def getSuperCategories(self):
        if self.superCategories != None:
            return self.categories
        
        annotations = self.loadAnnotationFile(typeData = 'train', typeAnnotation = "instances")

        superCategories = list(set([category['supercategory'] for category in annotations['categories']]))

        self.superCategories = superCategories
        return superCategories