## Semi-supervised Motion Style Transfer

<img src="./assets/111111.gif" alt="intro" style="zoom:75%;" />

### Installation

```
pip install -r requirements.txt
```



### Quick Start

```
python run.py --con ./m21_data/content/jumping-jack.bvh --sty ./m21_data/style/OLD-2.bvh
```

The original BVH file is saved to ./output/original/OLD-2_jumping-jack.bvh, 
while the version with removed foot sliding is saved to ./output/remove_fs/OLD-2_jumping-jack.bvh.

More examples:

```
python run.py --con ./m21_data/content/hop.bvh --sty ./m21_data/style/STR-1.bvh
```


```
python run.py --con ./m21_data/content/kick.bvh --sty ./m21_data/style/DEPRESSED.bvh
```


### Acknowledgments

The implementation of the graph neural network and the StyleNet in this repository is from the following repository:

https://github.com/DK-Jang/motion_puzzle

Also, we would like to thank the following teams for providing us with mocap data:

CMU dataset: http://mocap.cs.cmu.edu/  and  https://github.com/una-dinosauria/cmu-mocap

BFA dataset: https://github.com/DeepMotionEditing/deep-motion-editing

XIA dataset: https://github.com/DeepMotionEditing/deep-motion-editing and http://faculty.cs.tamu.edu/jchai/projects/SIG15/style-final.pdf
