# Detection-Segmentation-and-Feature-estimation-pipeline

The scripts are based on the official [label studio integration with fifty-one ](https://docs.voxel51.com/integrations/labelstudio.html).

## Annotation

First start Label studio:
``` 
label-studio start 
``` 

Then run the annotation script to automatically create a label studio project and start annotating a dataset.
``` 
python annotate.py
``` 

- `name`: A name for the dataset.
- `anno_key`: A name for the labelstudio project.
- `dataset_dir`: The path to the repository containing the dataset

See [here](https://docs.voxel51.com/integrations/labelstudio.html#label-studio-label-schema) for more details about the labeling schema.

- `label_field`: a string indicating a new or existing label field to annotate
- `label_type`: a string indicating the type of labels to annotate. 
  The possible label types are:
  - "classification": a single classification stored in Classification fields
  - "detections": object detections stored in Detections fields
  - "instances": instance segmentations stored in Detections fields with their mask attributes populated
  - "polylines": polylines stored in Polylines fields with their filled attributes set to False
  - "polygons": polygons stored in Polylines fields with their filled attributes set to True
  - "keypoints": keypoints stored in Keypoints fields
  - "segmentation": semantic segmentations stored in Segmentation fields

- `classes`: a list of strings indicating the class options for label_field or all fields in label_schema without classes specified.

To use Label studio, an API key must be provided, see [here](https://docs.voxel51.com/integrations/labelstudio.html#authentication).

## Augmentation

Run the annotation script to automatically augment a dataset using albumentation.
``` 
python annotate.py
```
- `name`: Name of the dataset to load.
- `json_file`: JSON file containing labels.
- `image_root`: The directory containing the images.
- `list_augmentations_file`: A file containing a list of albumentation augmmentations.
- `output_folder`: A folder to save the augmented images.
    
## Visualization

Inspect a dataset by running:
``` 
python augmented_fo_dataset.py
```
- `name`: Name of the dataset to load.
- `labels_path`: JSON file containing labels.
- `data_path`: The directory containing the images.

## Dataset preparation for Detectron

The script is based on the official [Detectron2 integration with fifty-one ](https://docs.voxel51.com/tutorials/detectron2.html).

Load the dataset, split it into training/validation and train a model.
``` 
python detectron_ready.py
```
