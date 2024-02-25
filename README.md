# panoramic
Create a panoramic image between two images, using basic Computer Vision Techniques.


## Installation

To run these games, you need to have Python installed on your system. Additionally, each game has its own set of dependencies which can be installed using `pip`.

```bash
pip install -r requirements.txt
```

## Usage

To create your own panoramic please change the config.ini file to have your images' paths to the file for image1 and image2.
Then run the `panoramic.py` file to generate a panoramic image of the two images. If the two images are not related, 
the code will throw an error saying the images cannot be made into a panoramic

## How it Works

### Corner detection

The first step in the code is a function that returns the corners in the individual images. For each image, a sobel operator is used to compute the gradient in the x and y directions of the image. Using these gradient images, we can create a second-moment matrix for each pixel around a fixed window. Finding the minimum eigenvalue of each second moment matrix, gives us an eigenimage of eigenvalues at each pixel. The larger eigenvalues represent "corners" in the image where the gradient shows change in both the x and y direction. We use the eigenvalue to account for rotation in the corner, so it does not only detect upright corners. The Non-Maximum Suppression takes the eigenimage, and suppresses values near each other so only one eigenvalue is picked in a group of pixels near each other that have large eigenvalues. We then pick up the top n eigenvalues as corners in the image. 

Example image:

<img src="im0.png" alt="Image of room with toys on tables, chairs, and a basket" width="300" height="200"/>

:arrow_down:

Example of Eigenimages:

<img src="eigenimage.png" alt="Eigenimage of the Image from above" width="300" height="200"/>


:arrow_down:


<img src="corners.png" alt="Image above with the detected corners" width="300" height="200"/>

