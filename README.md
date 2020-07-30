<img src="https://github.com/rajat-1994/DIF/blob/master/assets/banner.png" height="650" width="1280"/>

# Duplicate Image Finder
When creating image dataset for deep learning project, there are high changes that the dataset contains multiple duplicate images.\
This standalone tool will help in finding and removing those duplicate images using a simple interface. And if you are a developer
 you can easily customize the code according to your need.


# Table of contents

- [Duplicate Image Finder](#introduction)
- [Table of contents](#table-of-contents)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Reference](#reference)

# Demo
[(top)](#table-of-contents)

# Installation
[(top)](#table-of-contents)

1. ```git clone https://github.com/rajat-1994/DIF.git```
2. ```cd DIF```
3. ```pip install -r requirements.txt```

# Usage
[(top)](#table-of-contents)

Just run below command after installation and you are good to go.

* ```python app.py```

**NOTE** : As you delete images from the interface, in the backend a file `files.csv` is saved.
After you are done with cleaning your dataset you can just read the csv and filter the deleted images.

```python
df = pd.read_csv('files.csv')
df = df[df.is_deleted==0]        
```

# Reference
[(top)](#table-of-contents)

*[fastai Widgets](https://github.com/fastai/fastai)