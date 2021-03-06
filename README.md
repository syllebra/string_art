![Point Cloud String Art](results/edited/tiger_00.png)


# String Art Exploration

This project is an exploration of string art algorithms using python.


## Description

I explore multiple ways to create art from images using simulated sewing strings across various distributions of nails over a canvas

## Getting Started

### Dependencies
Create a python 3 environment (for exemple using conda) and install dependencies:
```sh
   $ pip install -r requirements.txt
   ```
### Executing program

* First, open string_art.py and edit the "INPUT PARAMETERS SECTION". Each parameter is well described in comments.
* Run the program
```sh
   $ python string_art.py
   ```
* The program will compute some cache data and start rendering
* you can pause the execution with any key while focusing on an image. Resume with any other key.
* At a given frequency, rendering images will be saved to "steps/[image_name]/run_***/[image_name_*******.png]"

## Layout Modes Samples
This sections shows some results with the different pin layout modes. More results (images and videos) can be found in the [results](https://github.com/syllebra/string_art/tree/main/results) directory.
### Point Cloud
![Point Cloud layout sample](results/sting_00_0017100.png)
![Point Cloud layout sample](results/lion_00_0003900.png)
<img src="results/elephant_00_0014880.png" alt="Point Cloud layout sample" width="50%"/><img src="results/joker_0006640.png" alt="Point Cloud layout sample" width="50%"/>

<center><img src="results/portrait_00_edited_0014240.png" alt="Point Cloud layout sample" width="70%"/></center>
<center><img src="results/nude_00_0009680.png" alt="Point Cloud layout sample" width="70%"/></center>

### Rectangle
![Rectangle layout sample](results/sa_manta_00.png)
![Rectangle layout sample](results/tribal_shark_0007800.png)
### Circle
<img src="results/renaud_circle.png" alt="Circle layout sample" width="40%"/><img src="results/pexels-george-desipris-818261_0007440.png" alt="Circle layout sample" width="40%"/>

### Perimeter
<img src="results/tribal_orca_00.jpg_0004800.png" alt="Perimeter layout sample" width="40%"/>

## "Invert" parameter sample
![Invert parameter sample](results/renaud_03_0048400.png)
![Invert parameter sample](results/renaud_03_0044640.png)
## Authors
C??dric Syllebranque
### Source photo
* [St??phane de Bourgies](https://www.bourgies.com/)
* [National Geographics](https://www.nationalgeographic.fr/)
* [Pexels](https://www.pexels.com)

## Version History

* 0.1
    * Initial Release

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgments

Inspiration, code snippets, etc.
* https://www.youtube.com/watch?v=RSRNZaq30W0
* https://www.youtube.com/watch?v=UsbBSttaJos
* [nailedit](https://github.com/hooyah/nailedit)



<!-- ROADMAP -->
## Roadmap

- [ ] Add Changelog
- [ ] Add json parameters
- [ ] Add Additional Templates w/ Examples

See the [open issues](https://github.com/syllebra/string_art/issues) for a full list of proposed features (and known issues).
