# Detection-Segmentation-and-Feature-estimation-pipeline

The scripts are based on the official [label studio](https://labelstud.io/) and [albumentation](https://albumentations.ai/) libraries.

## Annotation

First start Label studio:
``` 
label-studio start 
``` 
The possible label types which Label Studio supports are:
- "classification": a single classification stored in Classification fields
- "detections": object detections stored in Detections fields
- "instances": instance segmentations stored in Detections fields with their mask attributes populated
- "polylines": polylines stored in Polylines fields with their filled attributes set to False
- "polygons": polygons stored in Polylines fields with their filled attributes set to True
- "keypoints": keypoints stored in Keypoints fields
- "segmentation": semantic segmentations stored in Segmentation fields

Then create a template by integrating keypoints configurations to polygon configuration code.
Use below code to enable both keypoint and polygon annotations in Label studio
``` 
<View>
  <Header value="Select label and click the image to start"/>
  <Image name="image" value="$image" zoom="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="3" pointSize="small" opacity="0.9">
    <Label value="RockerArm_Object" background="red"/>
    <Label value="Bolt_Holes" background="green"/>
    ...
  </PolygonLabels>
  <KeyPointLabels name="kp-1" toName="image">
    <Label value="kp_rockerArm_object" background=" red" />
    <Label value="kp_rockerArm_target" background=" darkblue" />
  </KeyPointLabels>
</View>
``` 

## Augmentation

Run the annotation script to automatically augment a dataset using albumentation.
``` 
python scripts/pipelineNotebook.py
```

Here are the list of arguments required for successful augmentation.
- `json_file`: JSON file containing labels.
- `image_root`: The directory containing the images.
- `list_augmentations_file`: A file containing a list of albumentation augmmentations.
- `output_folder`: A folder to save the augmented images.
- `Augmentation`: accepts binary values True/False.
- `hasKeypoints`: Annotation information, if dataset contains keypoints: accepts binary values True/False.
- `hasBBox`: Annotation information, if dataset contains Bounding Box: accepts binary values True/False.
- `hasSeg`: Annotation information, if dataset contains Polygon Segmentation: accepts binary values True/False.
- `JSON_type`: accepts COCO or Custom. COCO to augment a COCO dataset, Custom to augments a fresh annotated dataset
- `Split`: accepts binary values True/False.
- `Val_Split`: Split percentage for the validation data: accepts value between 0-1.

Example:
```
python scripts/pipelineNotebook.py --json_file Dataset/AnnotatedJSON/AnnotatedJSON.json --image_root Dataset/AnnotatedImages --output_folder Dataset/aug --hasKeypoints True --hasBBox True --hasSeg True --Augmentation True --split False --Val_Split 0.5 --JSON_type Custom --list_augmentations_file notebook/Augmentations
```
