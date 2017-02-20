# machine_learning_review
This is a repository for developping the python codes and examples for a review paper on an <b> introduction to machine learning for physicists<b>. 
# Installing style sheet
Clone or download the repository. Then in the top directory use:
```
pip3 install .
```
You can now import the style sheet in any of your notebooks with the following:
```
import matplotlib as mpl
from ml_style import ml_style_1 as sty
mpl.rcParams.update(sty.style)
```
This will allow you to plot using the styles specified in ```ml_style/ml_style_1.py```.
# Updating style sheet
```
pip3 install . --upgrade
```